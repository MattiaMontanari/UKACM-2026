.usage as $usage
| {
    prompt_tokens: $usage.prompt_tokens,
    cached_tokens: (try $usage.prompt_tokens_details.cached_tokens catch null),
    uncached_prompt_tokens:
      (if ($usage.prompt_tokens | type) == "number"
          and ($usage.prompt_tokens_details.cached_tokens | type) == "number"
       then $usage.prompt_tokens - $usage.prompt_tokens_details.cached_tokens
       else null
       end),
    completion_tokens: $usage.completion_tokens,
    reasoning_tokens: (try $usage.completion_tokens_details.reasoning_tokens catch null),
    total_tokens: $usage.total_tokens
  }
