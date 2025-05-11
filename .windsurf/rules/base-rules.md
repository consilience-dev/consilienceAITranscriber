---
trigger: always_on
---

# Cardinal Rules for Development

## Be Incremental
Keep code changes small and focused. Favor many small commits over large, sweeping ones.

## Write Tests Early and Often
For every module or significant function, write unit or integration tests. Prioritize test coverage for core logic.

## Never Merge or Deploy with Failing Tests
Ensure all tests pass before merging to main or deploying. Broken pipelines are unacceptable.

## Prioritize Readability over Cleverness
Code should be easy to understand and modify. Leave comments explaining complex logic.

## Modularize Everything
Keep components loosely coupled and highly cohesive. Each module (e.g., transcription, diarization) should be testable and usable on its own.

## Design for GPU Local Use First
All processing code should be runnable and testable on a developer's local machine using available GPU resources (e.g., RTX 3070).

## Favor Open Standards and Formats
Use common, parsable formats (e.g., JSON, plain text) for input/output. Avoid vendor lock-in.

## Fail Loudly and Clearly
When errors happen, raise them with context. Don’t silently fail or return vague errors.

## Document Interfaces and Data Contracts
Clearly describe APIs, expected inputs/outputs, and how modules interact. Use README files or inline docstrings liberally.

## Wait for command prompt to finish
Always wait for any command line executions to finish and ensure you have all the output so that you can interpret it.

## Make sure the person you're chatting with understand all concepts.
Always take the time to make sure that the person you're talking to has a good understanding of all the concepts you're using.