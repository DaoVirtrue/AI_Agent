#!/usr/bin/env python3
"""
Phase 01: Environment Configuration Management
===============================================
使用 pydantic-settings 实现类型安全的多 Provider 配置管理。

特性：
- pydantic-settings 自动从 .env 文件读取
- SecretStr 防止 API Key 在日志中泄露
- field_validator 验证 API Key 格式
- Key 轮转（Primary + Fallback）支持
- .env.example 模板自动生成
- 启动时验证所有必需配置
- 打印安全报告（自动隐藏敏感信息）

用法：
    from environment_config import LLMConfig

    config = LLMConfig()  # 自动从 .env 加载

    # 获取 API Key
    key = config.get_openai_key()

    # 使用 config 值
    model = config.openai_default_model
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Literal, List, Dict, Any, ClassVar
from dataclasses import dataclass, field

from pydantic import (
    SecretStr,
    Field,
    field_validator,
    model_validator,
    ValidationInfo,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

# ============================================================================
# 常量定义
# ============================================================================

# 项目根目录（当前文件所在目录向上两级到 RAG 根目录）
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

# .env 文件路径
DEFAULT_ENV_PATH: Path = PROJECT_ROOT / ".env"

# 必需的 API Key 配置项（用于启动验证）
REQUIRED_KEYS_MAP: Dict[str, str] = {
    "openai_api_key": "OPENAI_API_KEY",
    "anthropic_api_key": "ANTHROPIC_API_KEY",
}


# ============================================================================
# 辅助类型
# ============================================================================

@dataclass
class KeyPair:
    """
    Primary + Fallback API Key 对。

    用途：
    - 当 Primary Key 达到速率限制时自动切换到 Fallback
    - 支持无缝 Key 轮转（rotation）
    - 记录每个 Key 的使用次数便于负载均衡

    Attributes:
        primary: 主 Key（SecretStr 确保不泄露）
        fallback: 备用 Key（可选）
        primary_usage: Primary Key 已使用次数
        fallback_usage: Fallback Key 已使用次数
    """
    primary: SecretStr
    fallback: Optional[SecretStr] = None
    primary_usage: int = 0
    fallback_usage: int = 0

    def get_active_key(self) -> SecretStr:
        """
        获取当前应使用的 Key。

        策略：优先使用 Primary，当 Primary 使用次数远超 Fallback 时
        自动切换到 Fallback 以均衡负载。
        """
        if self.fallback and self.primary_usage > self.fallback_usage * 2:
            return self.fallback
        return self.primary

    def record_usage(self, key_used: SecretStr) -> None:
        """
        记录一次 Key 使用。

        调用方在 API 调用后调用此方法来更新计数器。
        """
        if key_used.get_secret_value() == self.primary.get_secret_value():
            self.primary_usage += 1
        elif self.fallback and key_used.get_secret_value() == self.fallback.get_secret_value():
            self.fallback_usage += 1


# ============================================================================
# 主配置类
# ============================================================================

class LLMConfig(BaseSettings):
    """
    LLM 提供商的集中化配置管理。

    使用 pydantic-settings 实现：
    - 自动从 .env 文件读取环境变量
    - 类型验证（temperature 范围、retries 范围等）
    - 敏感信息保护（SecretStr 自动遮蔽日志输出）

    环境变量命名规范：
    - OPENAI_API_KEY: OpenAI API 密钥
    - OPENAI_DEFAULT_MODEL: 默认模型名
    - ANTHROPIC_API_KEY: Anthropic API 密钥
    - ANTHROPIC_DEFAULT_MODEL: 默认 Claude 模型
    - DEFAULT_TEMPERATURE: 默认温度参数
    - MAX_RETRIES: 最大重试次数
    - REQUEST_TIMEOUT: 请求超时（秒）
    - LOG_LEVEL: 日志级别

    Usage:
        # 方式 1: 自动加载项目根目录的 .env
        config = LLMConfig()

        # 方式 2: 指定 .env 路径
        config = LLMConfig(_env_file="/custom/path/.env")

        # 方式 3: 直接传参（测试用）
        config = LLMConfig(openai_api_key=SecretStr("sk-test"))
    """

    # pydantic-settings 配置
    model_config = SettingsConfigDict(
        env_file=str(DEFAULT_ENV_PATH),   # 自动加载的 .env 文件
        env_file_encoding="utf-8",        # .env 文件编码
        case_sensitive=False,             # 环境变量名不区分大小写
        extra="ignore",                   # 忽略未定义的额外环境变量
        validate_default=True,            # 验证默认值
    )

    # ========================
    # OpenAI 配置
    # ========================
    openai_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="OpenAI API 密钥 (sk-...)。可通过环境变量 OPENAI_API_KEY 设置。",
    )

    openai_fallback_key: Optional[SecretStr] = Field(
        default=None,
        description="OpenAI 备用 API Key。当主 Key 达到速率限制时自动切换。",
    )

    openai_default_model: str = Field(
        default="gpt-4o",
        description="默认使用的 OpenAI 模型。可通过环境变量 OPENAI_DEFAULT_MODEL 覆盖。",
    )

    openai_base_url: Optional[str] = Field(
        default=None,
        description="自定义 OpenAI API Base URL（用于代理或兼容 API）。例如 https://api.openai-proxy.com/v1",
    )

    openai_organization: Optional[str] = Field(
        default=None,
        description="OpenAI Organization ID（多组织账户时使用）。",
    )

    # ========================
    # Anthropic 配置
    # ========================
    anthropic_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Anthropic API 密钥 (sk-ant-...)。可通过环境变量 ANTHROPIC_API_KEY 设置。",
    )

    anthropic_fallback_key: Optional[SecretStr] = Field(
        default=None,
        description="Anthropic 备用 API Key。",
    )

    anthropic_default_model: str = Field(
        default="claude-sonnet-4-6-20250514",
        description="默认使用的 Claude 模型。可通过环境变量 ANTHROPIC_DEFAULT_MODEL 覆盖。",
    )

    anthropic_base_url: Optional[str] = Field(
        default=None,
        description="自定义 Anthropic API Base URL。",
    )

    # ========================
    # 通用配置
    # ========================
    default_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description=(
            "默认温度参数。0.0 = 最确定性（适合代码/事实），"
            "1.0 = 平衡，2.0 = 最随机（适合创意）。可通过 DEFAULT_TEMPERATURE 覆盖。"
        ),
    )

    max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="API 调用失败时的最大重试次数。可通过 MAX_RETRIES 覆盖。",
    )

    request_timeout: int = Field(
        default=60,
        ge=10,
        le=600,
        description="API 请求超时时间（秒）。可通过 REQUEST_TIMEOUT 覆盖。",
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="日志级别。可通过 LOG_LEVEL 覆盖。",
    )

    max_concurrent_requests: int = Field(
        default=10,
        ge=1,
        le=100,
        description="最大并发请求数（用于 asyncio.Semaphore 限流）。",
    )

    enable_cost_tracking: bool = Field(
        default=True,
        description="是否启用成本追踪。",
    )

    cost_warning_threshold: float = Field(
        default=1.0,
        ge=0.01,
        description="成本警告阈值（美元）。单次会话超过此金额时发出警告。",
    )

    # ========================
    # 向量数据库配置
    # ========================
    vector_db_type: Literal["chromadb", "qdrant", "pinecone", "milvus"] = Field(
        default="chromadb",
        description="使用的向量数据库类型。Phase 02 开始使用。",
    )

    vector_db_path: str = Field(
        default="data/vector_db",
        description="本地向量数据库的存储路径。",
    )

    chromadb_host: Optional[str] = Field(
        default=None,
        description="ChromaDB 服务器地址（客户端-服务器模式时使用）。",
    )

    chromadb_port: Optional[int] = Field(
        default=None,
        description="ChromaDB 服务器端口。",
    )

    # ========================
    # Embedding 配置
    # ========================
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="默认 Embedding 模型。",
    )

    embedding_dimension: int = Field(
        default=1536,
        ge=1,
        description="Embedding 向量维度。text-embedding-3-small: 1536, text-embedding-3-large: 3072。",
    )

    # ========================
    # 类变量（非配置项，运行时状态）
    # ========================
    _openai_key_pair: ClassVar[Optional[KeyPair]] = None
    _anthropic_key_pair: ClassVar[Optional[KeyPair]] = None
    _validated: ClassVar[bool] = False

    # =========================================================================
    # 验证器
    # =========================================================================

    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_key(cls, v: SecretStr) -> SecretStr:
        """
        验证 OpenAI API Key 格式。

        OpenAI Key 格式：
        - 标准 Key: sk-proj-... 或 sk-...
        - 如果为空字符串，表示未配置（启动时不会报错，但调用会失败）

        注意：这里只在 Key 非空时验证。空 Key 是合法的
        （允许只配置 Anthropic 不配置 OpenAI）。
        """
        key_value = v.get_secret_value()
        if key_value and not key_value.startswith("sk-"):
            raise ValueError(
                f"OpenAI API Key 必须以 'sk-' 开头。"
                f"当前值的前5个字符: '{key_value[:5]}...'"
            )
        return v

    @field_validator("anthropic_api_key")
    @classmethod
    def validate_anthropic_key(cls, v: SecretStr) -> SecretStr:
        """
        验证 Anthropic API Key 格式。

        Anthropic Key 格式：
        - 标准 Key: sk-ant-api03-... 或 sk-ant-...
        - 如果为空字符串，表示未配置
        """
        key_value = v.get_secret_value()
        if key_value and not key_value.startswith("sk-ant-"):
            raise ValueError(
                f"Anthropic API Key 必须以 'sk-ant-' 开头。"
                f"当前值的前5个字符: '{key_value[:5]}...'"
            )
        return v

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, v: str) -> str:
        """
        规范化日志级别输入。

        允许小写输入，自动转为大写。
        例如: "debug" → "DEBUG", "info" → "INFO"
        """
        if isinstance(v, str):
            return v.upper()
        return v

    @field_validator("embedding_dimension")
    @classmethod
    def validate_embedding_dimension(cls, v: int, info: ValidationInfo) -> int:
        """
        验证 Embedding 维度与模型匹配。

        自动检测不匹配的配置并发出警告。
        """
        model = info.data.get("embedding_model", "")
        expected = {
            "text-embedding-3-small": {512, 1536},
            "text-embedding-3-large": {256, 1024, 3072},
            "text-embedding-ada-002": {1536},
        }
        valid_dims = expected.get(str(model))
        if valid_dims and v not in valid_dims:
            logging.warning(
                f"Embedding 维度 {v} 与模型 '{model}' 不匹配。"
                f"有效维度: {valid_dims}。"
                f"如果使用 Matryoshka 表示可能仍然正确。"
            )
        return v

    @model_validator(mode="after")
    def ensure_at_least_one_provider(self):
        """
        确保至少配置了一个 LLM Provider。

        如果 OpenAI 和 Anthropic 的 Key 都为空，发出警告
        但不阻止启动（允许离线开发）。
        """
        openai_key = self.openai_api_key.get_secret_value()
        anthropic_key = self.anthropic_api_key.get_secret_value()
        if not openai_key and not anthropic_key:
            logging.warning(
                "⚠️  未配置任何 LLM Provider 的 API Key。\n"
                "  请在 .env 文件中设置 OPENAI_API_KEY 或 ANTHROPIC_API_KEY。\n"
                "  没有 API Key，任何 LLM 调用都会失败。"
            )
        return self

    # =========================================================================
    # Key 管理方法
    # =========================================================================

    def get_openai_key(self) -> Optional[str]:
        """
        获取 OpenAI API Key（明文）。

        Returns:
            API Key 字符串，如果未配置则返回 None。

        注意：
        - 返回值是明文 Key，不要打印到日志！
        - 在代码中使用后立即丢弃变量引用
        - 如果配置了 Fallback Key，使用 get_openai_key_pair() 获取
        """
        key = self.openai_api_key.get_secret_value()
        return key if key else None

    def get_anthropic_key(self) -> Optional[str]:
        """
        获取 Anthropic API Key（明文）。

        Returns:
            API Key 字符串，如果未配置则返回 None。
        """
        key = self.anthropic_api_key.get_secret_value()
        return key if key else None

    def get_openai_key_pair(self) -> Optional[KeyPair]:
        """
        获取 OpenAI Primary + Fallback Key 对。

        Returns:
            KeyPair 对象，支持自动 Key 轮转。

        Usage:
            pair = config.get_openai_key_pair()
            if pair:
                key = pair.get_active_key()
                pair.record_usage(key)
        """
        primary_key = self.openai_api_key.get_secret_value()
        if not primary_key:
            return None

        if self._openai_key_pair is None:
            fallback = (
                self.openai_fallback_key
                if self.openai_fallback_key
                and self.openai_fallback_key.get_secret_value()
                else None
            )
            self.__class__._openai_key_pair = KeyPair(
                primary=self.openai_api_key,
                fallback=fallback,
            )
        return self.__class__._openai_key_pair

    def get_anthropic_key_pair(self) -> Optional[KeyPair]:
        """
        获取 Anthropic Primary + Fallback Key 对。
        """
        primary_key = self.anthropic_api_key.get_secret_value()
        if not primary_key:
            return None

        if self._anthropic_key_pair is None:
            fallback = (
                self.anthropic_fallback_key
                if self.anthropic_fallback_key
                and self.anthropic_fallback_key.get_secret_value()
                else None
            )
            self.__class__._anthropic_key_pair = KeyPair(
                primary=self.anthropic_api_key,
                fallback=fallback,
            )
        return self.__class__._anthropic_key_pair

    # =========================================================================
    # 工具方法
    # =========================================================================

    def print_config(self, show_secrets: bool = False) -> None:
        """
        安全地打印当前配置。

        默认遮蔽所有敏感信息（API Key 只显示前几位）。

        Args:
            show_secrets: 如果为 True，显示完整 API Key（仅调试用！）。
                          生产环境永远不要传 True。

        Usage:
            config.print_config()           # 安全的遮蔽输出
            config.print_config(show_secrets=True)  # 仅调试！
        """
        print("=" * 60)
        print("📋 LLM Configuration Summary")
        print("=" * 60)

        def _mask(value: Optional[str], prefix_len: int = 8) -> str:
            """遮蔽敏感值"""
            if not value:
                return "<NOT SET>"
            if show_secrets:
                return value
            if len(value) <= prefix_len:
                return "*" * len(value)
            return value[:prefix_len] + "*" * (len(value) - prefix_len)

        # OpenAI
        openai_raw = self.get_openai_key()
        print(f"\n  [OpenAI]")
        print(f"    API Key:       {_mask(openai_raw)}")
        print(f"    Fallback Key:  {_mask(self.openai_fallback_key.get_secret_value() if self.openai_fallback_key else None)}")
        print(f"    Default Model: {self.openai_default_model}")
        print(f"    Base URL:      {self.openai_base_url or '(default)'}")
        print(f"    Organization:  {self.openai_organization or '(not set)'}")

        # Anthropic
        anthropic_raw = self.get_anthropic_key()
        print(f"\n  [Anthropic]")
        print(f"    API Key:       {_mask(anthropic_raw)}")
        print(f"    Fallback Key:  {_mask(self.anthropic_fallback_key.get_secret_value() if self.anthropic_fallback_key else None)}")
        print(f"    Default Model: {self.anthropic_default_model}")
        print(f"    Base URL:      {self.anthropic_base_url or '(default)'}")

        # General
        print(f"\n  [General]")
        print(f"    Temperature:     {self.default_temperature}")
        print(f"    Max Retries:     {self.max_retries}")
        print(f"    Timeout:         {self.request_timeout}s")
        print(f"    Log Level:       {self.log_level}")
        print(f"    Max Concurrent:  {self.max_concurrent_requests}")
        print(f"    Cost Tracking:   {'ON' if self.enable_cost_tracking else 'OFF'}")
        print(f"    Cost Warning:    ${self.cost_warning_threshold:.2f}")

        # Vector DB & Embedding
        print(f"\n  [Vector DB & Embedding]")
        print(f"    DB Type:         {self.vector_db_type}")
        print(f"    DB Path:         {self.vector_db_path}")
        print(f"    Embedding Model: {self.embedding_model}")
        print(f"    Embedding Dim:   {self.embedding_dimension}")

        print("\n" + "=" * 60)

        if not show_secrets:
            print("💡 提示: 使用 print_config(show_secrets=True) 查看完整 Key（仅调试）")

    def validate_startup(self) -> bool:
        """
        启动时验证所有必需的配置。

        在应用启动时调用，检查：
        1. 是否至少配置了一个 Provider 的 API Key
        2. 所有必需的 Key 格式是否正确
        3. 配置值是否在合理范围内

        Returns:
            True 如果所有验证通过。

        Raises:
            SystemExit: 如果关键配置缺失。

        Usage:
            config = LLMConfig()
            config.validate_startup()  # 在 main() 的第一行调用
        """
        errors: List[str] = []
        warnings: List[str] = []

        # 检查 Provider Key
        openai_key = self.get_openai_key()
        anthropic_key = self.get_anthropic_key()

        if not openai_key:
            warnings.append(
                "OPENAI_API_KEY 未设置。OpenAI 功能将不可用。"
            )
        if not anthropic_key:
            warnings.append(
                "ANTHROPIC_API_KEY 未设置。Anthropic 功能将不可用。"
            )
        if not openai_key and not anthropic_key:
            errors.append(
                "未配置任何 API Key！请设置 OPENAI_API_KEY 和/或 ANTHROPIC_API_KEY。\n"
                "  方式 1: 在项目根目录创建 .env 文件\n"
                "  方式 2: 设置系统环境变量\n"
                "  方式 3: 运行 python environment_config.py --generate-template 生成模板"
            )

        # 检查 temperature 范围
        if not (0.0 <= self.default_temperature <= 2.0):
            errors.append(f"Temperature 必须在 [0.0, 2.0] 之间，当前值: {self.default_temperature}")

        # 检查 timeout 合理性
        if self.request_timeout < 10:
            warnings.append(f"请求超时({self.request_timeout}s)较短，可能导致长文本生成失败")

        # 打印警告
        for w in warnings:
            logging.warning(f"  ⚠️  {w}")

        # 打印错误并退出
        if errors:
            print("\n" + "=" * 60)
            print("❌ 启动验证失败：")
            print("=" * 60)
            for e in errors:
                print(f"  🔴 {e}")
            print("=" * 60)
            print("\n💡 快速修复:")
            print(f"  cp {DEFAULT_ENV_PATH}.example {DEFAULT_ENV_PATH}")
            print(f"  然后编辑 {DEFAULT_ENV_PATH} 填入你的 API Key\n")
            self.__class__._validated = False
            return False

        self.__class__._validated = True
        return True

    def get_client_kwargs(self, provider: Literal["openai", "anthropic"]) -> Dict[str, Any]:
        """
        获取初始化 LLM Client 所需的关键字参数。

        这个方法的存在是为了统一 Provider 初始化模式：

        ```python
        config = LLMConfig()
        kwargs = config.get_client_kwargs("openai")
        # → {"api_key": "sk-...", "max_retries": 3, "timeout": 60}
        client = OpenAI(**kwargs)
        ```

        Args:
            provider: "openai" 或 "anthropic"

        Returns:
            可解包传入 Client 构造函数的参数字典
        """
        if provider == "openai":
            return {
                "api_key": self.get_openai_key() or "sk-placeholder",
                "max_retries": self.max_retries,
                "timeout": float(self.request_timeout),
                "base_url": self.openai_base_url,
                "organization": self.openai_organization,
            }
        elif provider == "anthropic":
            return {
                "api_key": self.get_anthropic_key() or "sk-ant-placeholder",
                "max_retries": self.max_retries,
                "timeout": float(self.request_timeout),
                "base_url": self.anthropic_base_url,
            }
        else:
            raise ValueError(f"Unknown provider: {provider}")

    @property
    def available_providers(self) -> List[str]:
        """
        返回当前已配置可用的 Provider 列表。

        Returns:
            如 ["openai", "anthropic"] 或 ["openai"] 等

        Usage:
            if "openai" in config.available_providers:
                use_openai()
        """
        providers = []
        if self.get_openai_key():
            providers.append("openai")
        if self.get_anthropic_key():
            providers.append("anthropic")
        return providers

    # =========================================================================
    # 静态工具方法
    # =========================================================================

    @staticmethod
    def generate_env_template(output_path: Optional[Path] = None) -> str:
        """
        生成 .env.example 模板文件。

        这个文件可以被复制为 .env 并填入真实 Key。
        模板中不包含任何敏感信息，可以安全地提交到 Git。

        Args:
            output_path: 输出路径。默认为 PROJECT_ROOT / ".env.example"

        Returns:
            生成的模板内容字符串

        Usage:
            python environment_config.py --generate-template
        """
        if output_path is None:
            output_path = PROJECT_ROOT / ".env.example"

        template_content = """# ============================================================================
# LLM API Configuration Template
# ============================================================================
# 使用说明:
#   1. 复制此文件: cp .env.example .env
#   2. 编辑 .env 填入你的真实 API Key
#   3. .env 文件已加入 .gitignore，不会被提交到 Git
#
# 获取 API Key:
#   OpenAI:  https://platform.openai.com/api-keys
#   Anthropic: https://console.anthropic.com/settings/keys
# ============================================================================

# ---- OpenAI ----
OPENAI_API_KEY=sk-your-openai-key-here
# OPENAI_FALLBACK_KEY=sk-your-fallback-key-here
# OPENAI_DEFAULT_MODEL=gpt-4o
# OPENAI_BASE_URL=https://api.openai.com/v1
# OPENAI_ORGANIZATION=org-xxxxxxxx

# ---- Anthropic ----
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
# ANTHROPIC_FALLBACK_KEY=sk-ant-your-fallback-key-here
# ANTHROPIC_DEFAULT_MODEL=claude-sonnet-4-6-20250514
# ANTHROPIC_BASE_URL=https://api.anthropic.com

# ---- General ----
# DEFAULT_TEMPERATURE=0.0
# MAX_RETRIES=3
# REQUEST_TIMEOUT=60
# LOG_LEVEL=INFO
# MAX_CONCURRENT_REQUESTS=10

# ---- Cost Tracking ----
# ENABLE_COST_TRACKING=true
# COST_WARNING_THRESHOLD=1.0

# ---- Vector DB ----
# VECTOR_DB_TYPE=chromadb
# VECTOR_DB_PATH=data/vector_db
# CHROMADB_HOST=localhost
# CHROMADB_PORT=8000

# ---- Embedding ----
# EMBEDDING_MODEL=text-embedding-3-small
# EMBEDDING_DIMENSION=1536
"""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(template_content, encoding="utf-8")
        print(f"✅ .env.example 模板已生成: {output_path}")
        return template_content


# ============================================================================
# 便捷单例
# ============================================================================

# 全局配置单例（惰性初始化）
_config_instance: Optional[LLMConfig] = None


def get_config() -> LLMConfig:
    """
    获取全局 LLMConfig 单例。

    在应用的任何地方调用 get_config() 获取同一个配置对象。

    Usage:
        from environment_config import get_config

        config = get_config()
        key = config.get_openai_key()
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = LLMConfig()
    return _config_instance


def reload_config(env_file: Optional[Path] = None) -> LLMConfig:
    """
    重新加载配置（用于运行时切换 .env 文件）。

    Args:
        env_file: 新的 .env 文件路径。None 则使用默认路径。

    Usage:
        config = reload_config(Path("/alt/.env"))
    """
    global _config_instance
    if env_file:
        _config_instance = LLMConfig(_env_file=str(env_file))
    else:
        _config_instance = LLMConfig()
    return _config_instance


# ============================================================================
# CLI 入口
# ============================================================================

if __name__ == "__main__":
    """
    命令行入口：验证配置或生成模板。

    用法：
        python environment_config.py                         # 加载并验证配置
        python environment_config.py --generate-template     # 生成 .env.example
        python environment_config.py --show-secrets          # 显示完整配置（含 Key）
        python environment_config.py --validate-only         # 仅验证，不打印配置
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="LLMConfig - 环境配置管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                          # 加载并显示配置
  %(prog)s --generate-template      # 生成 .env.example 模板
  %(prog)s --show-secrets           # 显示完整 Key（调试用）
  %(prog)s --validate-only          # 仅验证，无输出
        """.strip(),
    )
    parser.add_argument(
        "--generate-template",
        action="store_true",
        help=f"在 {DEFAULT_ENV_PATH.parent / '.env.example'} 生成 .env.example 模板",
    )
    parser.add_argument(
        "--show-secrets",
        action="store_true",
        help="显示完整的 API Key（仅调试！不要在生产中使用）",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="仅运行启动验证，不打印配置",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help=f"指定 .env 文件路径（默认: {DEFAULT_ENV_PATH}）",
    )

    args = parser.parse_args()

    # ---- 生成模板模式 ----
    if args.generate_template:
        LLMConfig.generate_env_template()
        sys.exit(0)

    # ---- 加载配置 ----
    print("🔧 加载配置...")

    if args.env_file:
        config = LLMConfig(_env_file=str(args.env_file))
        print(f"   使用自定义 .env: {args.env_file}")
    else:
        config = LLMConfig()
        print(f"   使用默认 .env: {DEFAULT_ENV_PATH}")

    # ---- 启动验证 ----
    if args.validate_only:
        success = config.validate_startup()
        if not success:
            sys.exit(1)
        print("✅ 配置验证通过")
        sys.exit(0)

    # ---- 常规模式：验证 + 显示 ----
    success = config.validate_startup()

    if not success:
        sys.exit(1)

    print("\n")
    config.print_config(show_secrets=args.show_secrets)

    print(f"\n✅ 配置加载成功")
    print(f"   可用 Provider: {config.available_providers}")
    print(f"   .env 路径: {DEFAULT_ENV_PATH}")
    print(f"   .env 存在: {DEFAULT_ENV_PATH.exists()}")

    # 提示下一步
    if not DEFAULT_ENV_PATH.exists():
        print(f"\n💡 .env 文件不存在，运行以下命令生成模板:")
        print(f"   python {Path(__file__).name} --generate-template")

    sys.exit(0)
