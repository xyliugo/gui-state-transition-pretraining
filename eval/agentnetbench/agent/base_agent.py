#!/usr/bin/env python3
import os
import base64
from abc import ABC, abstractmethod
import re
import asyncio
from typing import Dict, List, Tuple, Optional, Any
import concurrent.futures
import time

class BaseAgent(ABC):
    """Base class for trajectory-based agents."""
    
    # Class variables for client pool
    _client_pool = []
    _current_client_idx = 0
    _client_pool_initialized = False
    _base_url = None
    _api_key = None
    
    def __init__(self, model: str, client=None, base_url=None, api_key=None):
        """Initialize the base agent.
        
        Args:
            model: The model identifier to use for predictions
            client: Optional client object for making API calls
            base_url: Optional base URL for API calls (used if client is None)
            api_key: Optional API key for API calls (used if client is None)
        """
        self.model = model
        
        # Store credentials if provided
        if base_url:
            BaseAgent._base_url = base_url
        if api_key:
            BaseAgent._api_key = api_key
            
        # If client is None, create one using stored credentials
        if client is None:
            if BaseAgent._base_url and BaseAgent._api_key:
                from openai import AsyncOpenAI
                client = AsyncOpenAI(base_url=BaseAgent._base_url, api_key=BaseAgent._api_key)
            else:
                raise ValueError("Either client or base_url/api_key must be provided")
        
        self.client = client
        
        # Extract credentials from client if not already set
        if not BaseAgent._base_url or not BaseAgent._api_key:
            try:
                BaseAgent._base_url = client.base_url
                BaseAgent._api_key = client._api_key
            except Exception as e:
                print(f"Warning: Could not extract credentials from client: {e}")
                
        self.max_retries = 3
        self.max_tokens = 1000
        self.temperature = 0
        self.image_dir = None
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self.response_cache = {}
        self.last_request_time = 0
        self.min_request_interval = 0.1  # Minimum time between requests in seconds
        self.request_semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent requests
        
        # Global semaphore shared by all agent instances
        if not hasattr(BaseAgent, '_global_request_semaphore'):
            BaseAgent._global_request_semaphore = asyncio.Semaphore(10)  # Global limit of 10 requests across all instances
            
        # Initialize client pool once if not already done
        if not BaseAgent._client_pool_initialized:
            self._initialize_client_pool(client)
        # Add this client to the pool if not already there
        elif client not in BaseAgent._client_pool:
            BaseAgent._client_pool.append(client)
    
    def _initialize_client_pool(self, initial_client):
        """Initialize a pool of async clients using stored credentials when available."""
        BaseAgent._client_pool = []
        api_key_to_use = BaseAgent._api_key
        base_url_to_use = BaseAgent._base_url

        # Fallbacks from the initial client if possible
        if not api_key_to_use and hasattr(initial_client, 'api_key'):
            api_key_to_use = getattr(initial_client, 'api_key', None)
        if not api_key_to_use and hasattr(initial_client, '_api_key'):
            api_key_to_use = getattr(initial_client, '_api_key', None)
        if not base_url_to_use and hasattr(initial_client, 'base_url'):
            base_url_to_use = getattr(initial_client, 'base_url', None)

        # Last resort: environment
        if not api_key_to_use:
            api_key_to_use = os.environ.get('OPENAI_API_KEY')

        try:
            from openai import AsyncOpenAI
            for _ in range(10):
                if base_url_to_use and api_key_to_use:
                    new_client = AsyncOpenAI(base_url=base_url_to_use, api_key=api_key_to_use)
                elif base_url_to_use:
                    new_client = AsyncOpenAI(base_url=base_url_to_use)
                elif api_key_to_use:
                    new_client = AsyncOpenAI(api_key=api_key_to_use)
                else:
                    new_client = initial_client
                BaseAgent._client_pool.append(new_client)
            BaseAgent._client_pool_initialized = True
        except Exception:
            # Fallback: replicate the initial client
            BaseAgent._client_pool = [initial_client for _ in range(10)]
            BaseAgent._client_pool_initialized = True
    
    # Removed unused helpers: get_next_client, create, set_client_pool
    
    @abstractmethod
    def prompt(self, trajectory: Dict[str, Any], current_step: int) -> List[Dict[str, Any]]:
        """Generate prompt for the agent.
        
        Args:
            trajectory: The full trajectory data
            current_step: The current step index being processed
            
        Returns:
            List of message dictionaries for the model
        """
        pass

    @abstractmethod
    def parse_response(self, response: str, trajectory: Optional[Dict[str, Any]] = None, step_idx: Optional[int] = None) -> str:
        """Parse the raw response from the agent into executable form.
        
        Args:
            response: Raw response string from the model
            trajectory: Optional trajectory data for coordinate conversion
            step_idx: Optional step index for coordinate conversion
            
        Returns:
            Parsed response string ready for evaluation
        """
        pass

    @abstractmethod
    def extract_actions(self, action: str) -> List[Tuple[str, Any]]:
        """Extract individual actions from parsed response.
        
        Args:
            action: Parsed action string
            
        Returns:
            List of tuples containing (action_type, action_params)
            Example formats:
            - ('click', (x, y))
            - ('write', text)
            - ('press', [key1, key2])
            - ('terminate', status)
        """
        pass

    def load_image(self, image_file: str, image_dir: str) -> bytes:
        """Load image from file.
        
        Args:
            image_file: Name of the image file
            image_dir: Directory containing the image
            
        Returns:
            Image bytes
        """
        image_path = os.path.join(image_dir, image_file)
        with open(image_path, "rb") as f:
            return f.read()

    def predict(self, messages: List[Dict[str, Any]], instruction_prompt: str = None) -> Tuple[Optional[str], Optional[str]]:
        """Make prediction using the model.
        
        Args:
            messages: List of message dictionaries for the model
            instruction_prompt: Optional instruction prompt to return alongside response
            
        Returns:
            Tuple of (instruction_prompt, model_response) or (None, None) if failed
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return instruction_prompt, response.choices[0].message.content
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    return None, None

    async def predict_async(self, messages: List[Dict[str, Any]], instruction_prompt: str = None) -> Tuple[Optional[str], Optional[str]]:
        """Async version of predict method for concurrent requests.
        
        Args:
            messages: List of message dictionaries for the model
            instruction_prompt: Optional instruction prompt to return alongside response
            
        Returns:
            Tuple of (instruction_prompt, model_response) or (None, None) if failed
        """
        # Create a cache key for this request
        cache_key = str(hash(str(messages)))
        if cache_key in self.response_cache:
            return instruction_prompt, self.response_cache[cache_key]
            
        # Rate limiting to avoid overwhelming the API
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last_request)
            
        for attempt in range(self.max_retries):
            try:
                # Use both the global and instance semaphores to limit concurrent requests
                async with BaseAgent._global_request_semaphore:
                    # The instance semaphore is a backup in case global doesn't work
                    async with self.request_semaphore:
                        # Select which client to use for this request
                        client_pool_size = len(BaseAgent._client_pool)
                        if client_pool_size > 0:
                            # Get current index to use
                            client_idx = BaseAgent._current_client_idx
                            client_to_use = BaseAgent._client_pool[client_idx]
                            
                            # Advance index for next call
                            BaseAgent._current_client_idx = (BaseAgent._current_client_idx + 1) % client_pool_size
                        else:
                            client_idx = 0
                            client_to_use = self.client
                        
                        # Check if client is sync or async and handle accordingly
                        is_async_client = "async" in str(type(client_to_use)).lower()
                        
                        if is_async_client:
                            # For async clients, we can use await
                            response = await client_to_use.chat.completions.create(
                                model=self.model,
                                messages=messages,
                                temperature=self.temperature,
                                max_tokens=self.max_tokens,
                            )
                        else:
                            # For sync clients, we need to run in a thread pool
                            loop = asyncio.get_event_loop()
                            response = await loop.run_in_executor(
                                self.thread_pool,
                                lambda: client_to_use.chat.completions.create(
                                    model=self.model,
                                    messages=messages,
                                    temperature=self.temperature,
                                    max_tokens=self.max_tokens
                                )
                            )
                            
                        self.last_request_time = time.time()
                        result = response.choices[0].message.content
                        
                        # Cache the result
                        self.response_cache[cache_key] = result
                        
                        return instruction_prompt, result
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    return None, None

    async def test_traj_async(self, trajectory: Dict[str, Any], image_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """Async version of test_traj that makes concurrent requests for each step.
        
        Args:
            trajectory: The full trajectory data
            image_dir: Optional directory containing images (overrides default)
            
        Returns:
            List of results for each step
        """
        # Override default image_dir if provided
        if image_dir:
            self.image_dir = image_dir
            
        # Create tasks for all steps but process them with controlled concurrency
        results = [None] * len(trajectory['steps'])  # Pre-allocate results list
        
        async def process_step_with_semaphore(step_idx):
            # Generate prompt for current step
            messages = self.prompt(trajectory, step_idx)
            
            # Process step using semaphore
            result = await self._process_step_async(step_idx, messages, trajectory)
            results[step_idx] = result
        
        # Process steps with controlled concurrency
        tasks = []
        for step_idx in range(len(trajectory['steps'])):
            task = process_step_with_semaphore(step_idx)
            tasks.append(task)
            
        # Wait for all tasks to complete
        print(f"Processing trajectory {trajectory.get('task_id', 'unknown')} with {len(tasks)} steps")
        await asyncio.gather(*tasks)
        return results

    async def _process_step_async(self, step_idx: int, messages: List[Dict[str, Any]], trajectory: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a single step asynchronously.
        
        Args:
            step_idx: Index of the current step
            messages: List of message dictionaries for the model
            trajectory: Optional trajectory data for coordinate conversion
            
        Returns:
            Dictionary containing the step results
        """
        # Create a base result with raw response
        result = {
            'step_idx': step_idx,
            'instruction_prompt': None,
            'raw_response': None,
            'parsed_action': None,
            'actions': [],
            'parsing_error': False,
            'error_type': None,
            'error_message': None
        }
        
        # Get model response
        instruction_prompt, response = await self.predict_async(messages)
        result['instruction_prompt'] = instruction_prompt
        result['raw_response'] = response
        
        # If response is None, save the result with error info and return
        if response is None:
            print(f"Failed to get response for step {step_idx}")
            result['parsing_error'] = True
            result['error_type'] = 'no_response'
            return result
        
        # Try to parse response
        try:
            # Use run_in_executor to run parse_response in a thread pool
            loop = asyncio.get_event_loop()
            # Pass trajectory and step_idx to parse_response
            parsed_action = await loop.run_in_executor(
                self.thread_pool, 
                lambda: self.parse_response(response, trajectory, step_idx)
            )
            result['parsed_action'] = parsed_action
            
            # If parsed_action is None, mark as parsing error but continue
            if parsed_action is None:
                print(f"Failed to parse response for step {step_idx}")
                result['parsing_error'] = True
                result['error_type'] = 'parse_error'
                return result
            
            # Try to extract actions
            try:
                # Use run_in_executor to run extract_actions in a thread pool
                actions = await loop.run_in_executor(
                    self.thread_pool,
                    lambda: self.extract_actions(parsed_action)
                )
                result['actions'] = actions
                
                # If no actions extracted, mark as extraction error
                if not actions:
                    print(f"No actions extracted for step {step_idx}")
                    result['parsing_error'] = True
                    result['error_type'] = 'extraction_error'
                
            except Exception as e:
                print(f"Action extraction error for step {step_idx}: {e}")
                result['parsing_error'] = True
                result['error_type'] = 'extraction_exception'
                result['error_message'] = str(e)
            
        except Exception as e:
            print(f"Parsing error for step {step_idx}: {e}")
            result['parsing_error'] = True
            result['error_type'] = 'parse_exception'
            result['error_message'] = str(e)
        
        return result

    def test_traj(self, trajectory: Dict[str, Any], image_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """Test all steps in a trajectory.
        
        Args:
            trajectory: The full trajectory data
            image_dir: Optional directory containing images (overrides default)
            
        Returns:
            List of results for each step
        """
        # Override default image_dir if provided
        if image_dir:
            self.image_dir = image_dir
            
        results = []
        for step_idx in range(len(trajectory['steps'])):
            # Generate prompt for current step
            messages = self.prompt(trajectory, step_idx)
            
            # Get model response
            instruction_prompt, response = self.predict(messages)
            
            # Create a base result with raw response
            result = {
                'step_idx': step_idx,
                'instruction_prompt': instruction_prompt,
                'raw_response': response
            }
            
            # If response is None, save the result with error info and continue
            if response is None:
                print(f"Failed to get response for step {step_idx}")
                result['parsing_error'] = True
                result['error_type'] = 'no_response'
                result['parsed_action'] = None
                result['actions'] = []
                results.append(result)
                continue
            
            # Try to parse response
            try:
                parsed_action = self.parse_response(response, trajectory, step_idx)
                result['parsed_action'] = parsed_action
                
                # If parsed_action is None, mark as parsing error but continue
                if parsed_action is None:
                    print(f"Failed to parse response for step {step_idx}")
                    result['parsing_error'] = True
                    result['error_type'] = 'parse_error'
                    result['actions'] = []
                    results.append(result)
                    continue
                
                # Try to extract actions
                try:
                    actions = self.extract_actions(parsed_action)
                    result['actions'] = actions
                    
                    # If no actions extracted, mark as extraction error
                    if not actions:
                        print(f"No actions extracted for step {step_idx}")
                        result['parsing_error'] = True
                        result['error_type'] = 'extraction_error'
                    
                except Exception as e:
                    print(f"Action extraction error for step {step_idx}: {e}")
                    result['parsing_error'] = True
                    result['error_type'] = 'extraction_exception'
                    result['error_message'] = str(e)
                    result['actions'] = []
            
            except Exception as e:
                print(f"Parsing error for step {step_idx}: {e}")
                result['parsing_error'] = True
                result['error_type'] = 'parse_exception'
                result['error_message'] = str(e)
                result['parsed_action'] = None
                result['actions'] = []
            
            # Always append the result, regardless of parsing or extraction status
            results.append(result)
            
        return results 