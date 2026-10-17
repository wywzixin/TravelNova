import os

from openai import OpenAI
from typing import List, Dict, Any, Optional
from local_qwen import get_local_qwen_client
from observability import get_observability, safe_value

class LLMManager:
    """统一管理多个LLM模型的调用接口"""
    
    def __init__(self):
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")
        deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
        deepseek_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.local_qwen_client = None
        self.models = {
            "local-qwen": {
                "api_key": "local",
                "base_url": "local",
                "model_name": os.getenv("LOCAL_QWEN_MODEL_PATH", ""),
                "provider": "local_qwen"
            },
            "gpt-5": {
                "api_key": "",
                "base_url": "https://api.openai.com/v1",
                "model_name": "gpt-5-2025-08-07",
                "provider": "openai"
            },
            "claude-opus": {
                "api_key": "",
                "base_url": "https://api.anthropic.com/v1/",
                "model_name": "claude-opus-4-1-20250805",
                "provider": "anthropic"
            },
            "gemini-pro": {
                "api_key": "",
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                "model_name": "gemini-2.5-pro",
                "provider": "google",
                "extra_params": {"reasoning_effort": "low"}
            },
            "gemini-flash": {
                "api_key": "",
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                "model_name": "gemini-2.5-flash",
                "provider": "google",
                "extra_params": {"reasoning_effort": "low"}
            },
            "qwen-plus": {
                "api_key": "",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "model_name": "qwen-plus",
                "provider": "qwen"
            },
            "qwen-32b": {
                "api_key": "",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "model_name": "qwen3-32b",
                "provider": "qwen"
            },
            "qwen-14b": {
                "api_key": "",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "model_name": "qwen3-14b",
                "provider": "qwen"
            },
            "deepseek-v3": {
                "api_key": deepseek_api_key,
                "base_url": deepseek_base_url,
                "model_name": deepseek_model,
                "provider": "deepseek"
            },
            "deepseek-chat": {
                "api_key": deepseek_api_key,
                "base_url": deepseek_base_url,
                "model_name": deepseek_model,
                "provider": "deepseek"
            },
            "deepseek-r1": {
                "api_key": deepseek_api_key,
                "base_url": deepseek_base_url,
                "model_name": os.getenv("DEEPSEEK_REASONER_MODEL", "deepseek-reasoner"),
                "provider": "deepseek",
                "enable_thinking": True
            },
            "glm-4.5": {
                "api_key": "",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "model_name": "glm-4.5",
                "provider": "glm",
                "enable_thinking": True
            }
        }

    def get_available_models(self) -> List[str]:
        """获取所有可用模型的名称列表"""
        return list(self.models.keys())

    def call_model(self, model_name: str, prompt: str = None, system_prompt: Optional[str] = None, messages: Optional[List[Dict]] = None, tools: Optional[List[Dict]] = None, tool_call_handler = None, stream: bool = False, show_thinking: bool = True) -> str:
        """Call a configured model and record the call in Langfuse."""
        model_config = self.models.get(model_name, {})
        with get_observability().observation(
            name=f"llm.{model_name}",
            as_type="generation",
            input={
                "prompt": prompt,
                "system_prompt": system_prompt,
                "messages": messages,
                "tools": tools,
                "stream": stream,
            },
            metadata={"component": "llm_main"},
            model=model_config.get("model_name"),
        ) as generation:
            try:
                result = self._call_model_impl(
                    model_name=model_name,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    messages=messages,
                    tools=tools,
                    tool_call_handler=tool_call_handler,
                    stream=stream,
                    show_thinking=show_thinking,
                )
                generation.update(output=safe_value(result))
                return result
            except Exception as exc:
                generation.update(
                    level="ERROR",
                    status_message=str(exc),
                    output={"error": str(exc)},
                )
                raise

    def _call_model_impl(self, model_name: str, prompt: str = None, system_prompt: Optional[str] = None, messages: Optional[List[Dict]] = None, tools: Optional[List[Dict]] = None, tool_call_handler = None, stream: bool = False, show_thinking: bool = True) -> str:
        """
        调用指定模型进行推理
        
        Args:
            model_name: 模型名称
            prompt: 用户输入的提示（与messages参数二选一）
            system_prompt: 系统提示（可选）
            messages: 消息列表（与prompt参数二选一，优先使用此参数）
            tools: 工具定义列表（可选）
            tool_call_handler: 工具调用处理函数（可选）
            stream: 是否使用流式输出
            show_thinking: 是否显示思考过程（仅支持思考模型）
        
        Returns:
            str: 模型的响应内容
        """
        if model_name not in self.models:
            raise ValueError(f"模型 '{model_name}' 不存在。可用模型: {self.get_available_models()}")
        
        model_config = self.models[model_name]

        if model_config["provider"] == "local_qwen":
            if self.local_qwen_client is None:
                self.local_qwen_client = get_local_qwen_client()
            if messages is None:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                if prompt:
                    messages.append({"role": "user", "content": prompt})
            response = self.local_qwen_client.chat.completions.create(
                model="qwen3-0.6b-local",
                messages=messages,
                tools=tools,
            )
            return response.choices[0].message.content or ""
        
        # 构建消息
        if messages is not None:
            # 使用传入的messages列表
            final_messages = messages
        else:
            # 使用传统的prompt方式
            final_messages = []
            if system_prompt:
                final_messages.append({"role": "system", "content": system_prompt})
            if prompt:
                final_messages.append({"role": "user", "content": prompt})
        
        # 创建客户端
        client = OpenAI(
            api_key=model_config["api_key"],
            base_url=model_config["base_url"]
        )
        
        # 根据模型类型决定调用方式
        if model_config["provider"] in ["openai", "anthropic", "google"]:
            return self._call_standard_model(client, model_config, final_messages, tools, tool_call_handler, stream)
        elif model_config["provider"] in ["qwen"]:
            return self._call_qwen_model(client, model_config, final_messages, tools, tool_call_handler, stream)
        elif model_config["provider"] in ["deepseek", "glm"] and model_config.get("enable_thinking", False):
            return self._call_thinking_model(client, model_config, final_messages, stream, show_thinking)
        else:
            return self._call_standard_model(client, model_config, final_messages, tools, tool_call_handler, stream)

    def _call_standard_model(self, client: OpenAI, model_config: Dict, messages: List[Dict], tools: Optional[List[Dict]], tool_call_handler, stream: bool) -> str:
        """调用标准模型（GPT, Claude, Gemini等）"""
        try:
            extra_params = model_config.get("extra_params", {})
            
            if stream:
                completion = client.chat.completions.create(
                    model=model_config["model_name"],
                    messages=messages,
                    stream=True,
                    **extra_params
                )
                
                content_parts = []
                for chunk in completion:
                    if chunk.choices:
                        content = chunk.choices[0].delta.content or ""
                        print(content, end="", flush=True)
                        content_parts.append(content)
                
                print()
                return "".join(content_parts)
            else:
                response = client.chat.completions.create(
                    model=model_config["model_name"],
                    messages=messages,
                    **extra_params
                )
                return response.choices[0].message.content
                
        except Exception as e:
            return f"调用模型时出错: {e}"

    def _call_qwen_model(self, client: OpenAI, model_config: Dict, messages: List[Dict], tools: Optional[List[Dict]], tool_call_handler, stream: bool) -> str:
        """调用Qwen模型"""
        try:
            if stream:
                completion = client.chat.completions.create(
                    model=model_config["model_name"],
                    messages=messages,
                    stream=True
                )
                
                content_parts = []
                print("AI: ", end="", flush=True)
                
                for chunk in completion:
                    if chunk.choices:
                        content = chunk.choices[0].delta.content or ""
                        print(content, end="", flush=True)
                        content_parts.append(content)
                
                print()
                return "".join(content_parts)
            else:
                response = client.chat.completions.create(
                    model=model_config["model_name"],
                    messages=messages
                )
                return response.choices[0].message.content
                
        except Exception as e:
            return f"调用Qwen模型时出错: {e}"

    def _call_thinking_model(self, client: OpenAI, model_config: Dict, messages: List[Dict], stream: bool, show_thinking: bool) -> str:
        """调用支持思考过程的模型（DeepSeek, GLM等）"""
        try:
            request_kwargs = {
                "model": model_config["model_name"],
                "messages": messages,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            if model_config.get("provider") not in ["deepseek"]:
                request_kwargs["extra_body"] = {"enable_thinking": True}
            completion = client.chat.completions.create(**request_kwargs)

            reasoning_content = ""
            answer_content = ""
            is_answering = False
            
            if show_thinking:
                print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")

            for chunk in completion:
                if not chunk.choices:
                    if show_thinking:
                        print("\n" + "="*20+"Usage"+"="*20)
                        print(chunk.usage)
                    continue
                    
                delta = chunk.choices[0].delta

                if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                    if show_thinking and not is_answering:
                        print(delta.reasoning_content, end="", flush=True)
                    reasoning_content += delta.reasoning_content

                if hasattr(delta, "content") and delta.content:
                    if show_thinking and not is_answering:
                        print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                        is_answering = True
                    if show_thinking:
                        print(delta.content, end="", flush=True)
                    answer_content += delta.content

            if show_thinking:
                print()
            
            return f"[思考]\n{reasoning_content}\n\n[结果]\n{answer_content}" if reasoning_content else answer_content
            
        except Exception as e:
            return f"调用思考模型时出错: {e}"


# 全局实例
llm_manager = LLMManager()

def call_llm(model_name: str, prompt: str = None, system_prompt: Optional[str] = None, messages: Optional[List[Dict]] = None, stream: bool = False, show_thinking: bool = True) -> str:
    """
    调用指定LLM模型
    
    Args:
        model_name: 模型名称
        prompt: 用户输入（与messages二选一）
        system_prompt: 系统提示（可选）
        messages: 消息列表（与prompt二选一，优先使用此参数）
        stream: 是否流式输出
        show_thinking: 是否显示思考过程
        
    Returns:
        str: 模型响应
    """
    return llm_manager.call_model(
        model_name=model_name,
        prompt=prompt,
        system_prompt=system_prompt,
        messages=messages,
        stream=stream,
        show_thinking=show_thinking,
    )

def get_model_list() -> List[str]:
    """获取所有可用模型列表"""
    return llm_manager.get_available_models()

# 示例用法
if __name__ == "__main__":
    # 获取模型列表
    print("可用模型:", get_model_list())
    
    # 调用示例
    response = call_llm("gpt-5", "你好，请介绍一下自己")
    print(response)
    
    
    
