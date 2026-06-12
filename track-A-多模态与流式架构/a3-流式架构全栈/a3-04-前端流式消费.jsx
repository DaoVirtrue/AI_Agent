/**
 * A3-04: 前端流式消费组件 (Frontend Streaming Consumer)
 * ==========================================================
 * React组件库: 基于EventSource的SSE流式消费
 *
 * 功能:
 *   1. useState 累积接收token
 *   2. 断线重连 + 指数退避
 *   3. 进度条显示
 *   4. Markdown实时渲染
 *   5. 取消/中断支持
 *
 * 用法:
 *   <StreamingChat url="/api/chat/stream" />
 *
 * 依赖: react, react-markdown, remark-gfm
 *   npm install react-markdown remark-gfm
 */

import React, {
  useState,
  useEffect,
  useRef,
  useCallback,
  useMemo,
} from "react";

// ============================================================
// Section 1: useEventSource Hook — SSE连接管理
// ============================================================

/**
 * useEventSource Hook
 *
 * 管理SSE连接的生命周期:
 * - 连接建立
 * - 自动重连（指数退避）
 * - 事件监听
 * - 连接关闭
 *
 * @param {string} url - SSE端点URL
 * @param {Object} options - 配置项
 * @returns {Object} { events, connectionState, retryCount, close }
 */
function useEventSource(url, options = {}) {
  const {
    enabled = true,
    maxRetries = 5,
    baseDelay = 1000,       // 基础重连延迟ms
    maxDelay = 30000,       // 最大重连延迟ms
    onToken = null,
    onDone = null,
    onError = null,
    method = "GET",         // GET(EventSource) | POST(fetch+readableStream)
    body = null,
  } = options;

  const [connectionState, setConnectionState] = useState("disconnected");
  // "disconnected" | "connecting" | "connected" | "reconnecting" | "error"
  const [retryCount, setRetryCount] = useState(0);
  const eventSourceRef = useRef(null);
  const retryTimerRef = useRef(null);
  const mountedRef = useRef(true);

  const close = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    if (retryTimerRef.current) {
      clearTimeout(retryTimerRef.current);
      retryTimerRef.current = null;
    }
    setConnectionState("disconnected");
  }, []);

  useEffect(() => {
    mountedRef.current = true;

    const connect = () => {
      if (!enabled || !mountedRef.current) return;

      setConnectionState((prev) =>
        prev === "reconnecting" ? "reconnecting" : "connecting"
      );

      // POST模式的SSE（用于需要请求体的场景）
      if (method === "POST" && body) {
        connectViaFetch();
      } else {
        connectViaEventSource();
      }
    };

    const connectViaEventSource = () => {
      const es = new EventSource(url);
      eventSourceRef.current = es;

      es.onopen = () => {
        if (mountedRef.current) {
          setConnectionState("connected");
          setRetryCount(0); // 重置重试计数
        }
      };

      // 监听自定义事件
      es.addEventListener("token", (e) => {
        try {
          const data = JSON.parse(e.data);
          onToken?.(data);
        } catch (err) {
          console.error("解析token事件失败:", err);
        }
      });

      es.addEventListener("done", (e) => {
        try {
          const data = JSON.parse(e.data);
          onDone?.(data);
        } catch (err) {
          onDone?.({ finish_reason: "stop" });
        }
        es.close();
        setConnectionState("disconnected");
      });

      es.addEventListener("error", (e) => {
        // EventSource的error事件在连接失败和服务器关闭时都会触发
        if (es.readyState === EventSource.CLOSED) {
          onError?.(new Error("连接已关闭"));
          handleReconnect();
        } else {
          console.warn("SSE连接错误 (尝试重连中...)");
        }
      });

      es.onerror = () => {
        // 浏览器会自动重连，但我们也做手动重连
      };
    };

    const connectViaFetch = async () => {
      try {
        setConnectionState("connecting");

        const response = await fetch(url, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "text/event-stream",
          },
          body: JSON.stringify(body),
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        setConnectionState("connected");
        setRetryCount(0);

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || ""; // 保留不完整的行

          let currentEvent = null;
          for (const line of lines) {
            if (line.startsWith("event: ")) {
              currentEvent = line.slice(7).trim();
            } else if (line.startsWith("data: ") && currentEvent) {
              const data = line.slice(6);
              try {
                const parsed = JSON.parse(data);
                if (currentEvent === "token") onToken?.(parsed);
                else if (currentEvent === "done") {
                  onDone?.(parsed);
                  setConnectionState("disconnected");
                  return;
                }
              } catch {
                // raw data
                if (currentEvent === "token") onToken?.({ token: data });
              }
              currentEvent = null;
            }
          }
        }
      } catch (err) {
        console.error("Fetch SSE失败:", err);
        onError?.(err);
        handleReconnect();
      }
    };

    const handleReconnect = () => {
      if (!mountedRef.current || retryCount >= maxRetries) {
        setConnectionState("error");
        return;
      }

      setConnectionState("reconnecting");
      setRetryCount((prev) => prev + 1);

      // 指数退避: delay = min(baseDelay * 2^retry, maxDelay)
      const delay = Math.min(
        baseDelay * Math.pow(2, retryCount),
        maxDelay
      );
      // 添加抖动 (±20%)
      const jitter = delay * 0.2 * (Math.random() * 2 - 1);
      const finalDelay = delay + jitter;

      console.log(`[重连] ${(finalDelay / 1000).toFixed(1)}s后第${retryCount + 1}次重试`);

      retryTimerRef.current = setTimeout(() => {
        if (mountedRef.current) {
          connect();
        }
      }, finalDelay);
    };

    connect();

    return () => {
      mountedRef.current = false;
      close();
    };
  }, [url, enabled]);

  // 当url变化时重连
  useEffect(() => {
    return () => {
      mountedRef.current = false;
      close();
    };
  }, []);

  return { connectionState, retryCount, close };
}


// ============================================================
// Section 2: StreamingChat 组件 — 主组件
// ============================================================

/**
 * StreamingChat 组件
 *
 * 完整功能的流式聊天组件
 */
function StreamingChat({
  url = "/api/chat/stream",
  method = "POST",
  placeholder = "输入您的问题...",
  className = "",
}) {
  const [messages, setMessages] = useState([]);      // 历史消息
  const [currentText, setCurrentText] = useState(""); // 当前流式文本
  const [isStreaming, setIsStreaming] = useState(false);
  const [inputValue, setInputValue] = useState("");
  const [error, setError] = useState(null);
  const [requestBody, setRequestBody] = useState(null);

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // 自动滚动到底部
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    if (isStreaming) scrollToBottom();
  }, [currentText, isStreaming]);

  // 处理token回调
  const handleToken = useCallback((data) => {
    if (data.token) {
      setCurrentText((prev) => prev + data.token);
    }
  }, []);

  // 处理完成回调
  const handleDone = useCallback((data) => {
    setCurrentText((prev) => {
      if (prev.trim()) {
        setMessages((msgs) => [
          ...msgs,
          {
            role: "assistant",
            content: prev,
            timestamp: Date.now(),
            finishReason: data?.finish_reason || "stop",
          },
        ]);
      }
      return "";
    });
    setIsStreaming(false);
    setRequestBody(null);
  }, []);

  // 处理错误回调
  const handleError = useCallback((err) => {
    setError(err.message || "连接错误");
    setIsStreaming(false);
    setRequestBody(null);
  }, []);

  // SSE Hook（仅在streaming且有请求体时启用）
  const { connectionState, retryCount, close } = useEventSource(url, {
    enabled: isStreaming && requestBody !== null,
    method,
    body: requestBody,
    maxRetries: 3,
    onToken: handleToken,
    onDone: handleDone,
    onError: handleError,
  });

  // 发送消息
  const handleSend = useCallback(() => {
    const trimmed = inputValue.trim();
    if (!trimmed || isStreaming) return;

    // 添加用户消息
    setMessages((prev) => [
      ...prev,
      {
        role: "user",
        content: trimmed,
        timestamp: Date.now(),
      },
    ]);

    setInputValue("");
    setCurrentText("");
    setError(null);
    setIsStreaming(true);
    setRequestBody({ query: trimmed, stream: true });
  }, [inputValue, isStreaming]);

  // 取消生成
  const handleCancel = useCallback(() => {
    close();
    // 保留已生成的文本
    if (currentText.trim()) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: currentText + " [已中断]",
          timestamp: Date.now(),
          interrupted: true,
        },
      ]);
    }
    setCurrentText("");
    setIsStreaming(false);
    setRequestBody(null);
  }, [currentText, close]);

  // 键盘事件
  const handleKeyDown = useCallback(
    (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend]
  );

  return (
    <div className={`streaming-chat ${className}`} style={styles.container}>
      {/* 消息列表 */}
      <div style={styles.messagesContainer}>
        {messages.length === 0 && !isStreaming && (
          <div style={styles.emptyState}>
            <p>开始对话，体验流式RAG回答</p>
          </div>
        )}

        {messages.map((msg, idx) => (
          <MessageBubble key={idx} message={msg} />
        ))}

        {/* 当前流式输出 */}
        {isStreaming && currentText && (
          <MessageBubble
            message={{
              role: "assistant",
              content: currentText,
              streaming: true,
            }}
          />
        )}

        {/* 连接状态指示器 */}
        {isStreaming && (
          <ConnectionIndicator
            state={connectionState}
            retryCount={retryCount}
          />
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* 错误提示 */}
      {error && (
        <div style={styles.error}>
          <span>{error}</span>
          <button onClick={() => setError(null)} style={styles.errorDismiss}>
            x
          </button>
        </div>
      )}

      {/* 输入区 */}
      <div style={styles.inputContainer}>
        <textarea
          ref={inputRef}
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={isStreaming}
          style={styles.textarea}
          rows={2}
        />
        <div style={styles.buttonGroup}>
          {isStreaming ? (
            <button onClick={handleCancel} style={styles.cancelButton}>
              中断
            </button>
          ) : (
            <button
              onClick={handleSend}
              disabled={!inputValue.trim()}
              style={{
                ...styles.sendButton,
                opacity: inputValue.trim() ? 1 : 0.5,
              }}
            >
              发送
            </button>
          )}
        </div>
      </div>
    </div>
  );
}


// ============================================================
// Section 3: 子组件
// ============================================================

/**
 * MessageBubble — 消息气泡
 * 支持Markdown渲染
 */
function MessageBubble({ message }) {
  const isUser = message.role === "user";

  // 尝试懒加载react-markdown
  const [MarkdownRenderer, setMarkdownRenderer] = useState(null);

  useEffect(() => {
    let cancelled = false;
    import("react-markdown")
      .then((mod) => {
        if (!cancelled) setMarkdownRenderer(() => mod.default);
      })
      .catch(() => {
        // react-markdown未安装，使用纯文本
      });
    return () => { cancelled = true; };
  }, []);

  const renderContent = () => {
    if (MarkdownRenderer && !isUser) {
      try {
        return <MarkdownRenderer remarkPlugins={[]}>{message.content}</MarkdownRenderer>;
      } catch {
        // fall through
      }
    }
    return <p style={{ whiteSpace: "pre-wrap" }}>{message.content}</p>;
  };

  return (
    <div
      style={{
        ...styles.messageBubble,
        ...(isUser ? styles.userBubble : styles.assistantBubble),
      }}
    >
      <div style={styles.messageRole}>
        {isUser ? "You" : "RAG Assistant"}
        {message.interrupted && " (已中断)"}
      </div>
      <div style={styles.messageContent}>
        {renderContent()}
        {message.streaming && <span style={styles.cursor}>|</span>}
      </div>
      <div style={styles.messageTime}>
        {new Date(message.timestamp).toLocaleTimeString()}
      </div>
    </div>
  );
}


/**
 * ConnectionIndicator — 连接状态指示器
 */
function ConnectionIndicator({ state, retryCount }) {
  const stateConfig = {
    connecting: { text: "连接中...", color: "#FFA500" },
    connected: { text: "生成中...", color: "#4CAF50" },
    reconnecting: {
      text: `重连中 (${retryCount})...`,
      color: "#FF9800",
    },
    error: { text: "连接失败", color: "#F44336" },
  };

  const config = stateConfig[state] || { text: state, color: "#999" };

  return (
    <div style={{ ...styles.indicator, borderLeftColor: config.color }}>
      <span style={{ color: config.color }}>{config.text}</span>
      {state === "connected" && <ProgressBar />}
    </div>
  );
}


/**
 * ProgressBar — 动画进度条
 */
function ProgressBar() {
  return (
    <div style={styles.progressBarContainer}>
      <div style={styles.progressBarFill} />
    </div>
  );
}

// ============================================================
// Section 4: 样式
// ============================================================

const styles = {
  container: {
    display: "flex",
    flexDirection: "column",
    height: "100vh",
    maxWidth: "800px",
    margin: "0 auto",
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    background: "#f5f5f5",
  },
  messagesContainer: {
    flex: 1,
    overflowY: "auto",
    padding: "20px",
  },
  emptyState: {
    textAlign: "center",
    color: "#999",
    paddingTop: "100px",
  },
  messageBubble: {
    marginBottom: "16px",
    padding: "12px 16px",
    borderRadius: "12px",
    maxWidth: "80%",
    wordBreak: "break-word",
  },
  userBubble: {
    marginLeft: "auto",
    background: "#1976D2",
    color: "white",
  },
  assistantBubble: {
    marginRight: "auto",
    background: "white",
    color: "#333",
    boxShadow: "0 1px 3px rgba(0,0,0,0.12)",
  },
  messageRole: {
    fontSize: "12px",
    fontWeight: "bold",
    marginBottom: "4px",
    opacity: 0.7,
  },
  messageContent: {
    fontSize: "15px",
    lineHeight: "1.6",
  },
  messageTime: {
    fontSize: "11px",
    opacity: 0.5,
    marginTop: "6px",
  },
  cursor: {
    display: "inline-block",
    animation: "blink 1s step-end infinite",
    fontWeight: "bold",
    color: "#1976D2",
  },
  inputContainer: {
    padding: "16px",
    background: "white",
    borderTop: "1px solid #e0e0e0",
    display: "flex",
    gap: "8px",
  },
  textarea: {
    flex: 1,
    padding: "10px",
    borderRadius: "8px",
    border: "1px solid #ccc",
    fontSize: "14px",
    resize: "none",
    fontFamily: "inherit",
  },
  buttonGroup: {
    display: "flex",
    flexDirection: "column",
    justifyContent: "flex-end",
  },
  sendButton: {
    padding: "10px 20px",
    background: "#1976D2",
    color: "white",
    border: "none",
    borderRadius: "8px",
    cursor: "pointer",
    fontSize: "14px",
    fontWeight: "bold",
  },
  cancelButton: {
    padding: "10px 20px",
    background: "#F44336",
    color: "white",
    border: "none",
    borderRadius: "8px",
    cursor: "pointer",
    fontSize: "14px",
  },
  error: {
    padding: "10px 16px",
    background: "#FFEBEE",
    color: "#C62828",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
  },
  errorDismiss: {
    background: "none",
    border: "none",
    cursor: "pointer",
    fontSize: "16px",
    color: "#C62828",
  },
  indicator: {
    padding: "8px 16px",
    borderLeft: "3px solid",
    marginBottom: "8px",
    fontSize: "13px",
  },
  progressBarContainer: {
    width: "100%",
    height: "3px",
    background: "#e0e0e0",
    borderRadius: "2px",
    marginTop: "4px",
    overflow: "hidden",
  },
  progressBarFill: {
    height: "100%",
    width: "30%",
    background: "#4CAF50",
    borderRadius: "2px",
    animation: "progressMove 0.8s ease-in-out infinite",
    "@keyframes progressMove": {
      "0%": { transform: "translateX(-100%)" },
      "100%": { transform: "translateX(400%)" },
    },
  },
};


// ============================================================
// Section 5: Demo应用入口
// ============================================================

/**
 * App — 演示应用
 */
function App() {
  return (
    <div>
      <h1 style={{ textAlign: "center", padding: "20px", color: "#333" }}>
        RAG 流式聊天演示
      </h1>
      <StreamingChat
        url="/api/chat/stream"
        method="POST"
        placeholder="输入您的问题，体验流式RAG..."
      />
    </div>
  );
}

// 注意: 在实际React应用中，使用以下方式渲染:
//   import { createRoot } from 'react-dom/client';
//   const root = createRoot(document.getElementById('root'));
//   root.render(<App />);

export default StreamingChat;
export { useEventSource, StreamingChat, App };
