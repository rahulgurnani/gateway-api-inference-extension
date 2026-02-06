   1. Verify the `HttpRoute` status:

      Check that the HTTPRoute was successfully configured and references were resolved. Example command for vllm-llama3-8b-instruct:


      ```bash
      kubectl get httproute vllm-llama3-8b-instruct -o yaml
      ```

      The `HttpRoute` status should include `Accepted=True` and `ResolvedRefs=True`.

   1. Verify the `InferencePool` Status:

      Make sure the `InferencePool` is active before sending traffic. Example command for vllm-llama3-8b-instruct:


      ```bash
      kubectl get inferencepool vllm-llama3-8b-instruct -o yaml
      ```

      The `InferencePool` status should include `Accepted=True` and `ResolvedRefs=True`.
