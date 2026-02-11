<template>
  <section class="card">
    <div class="messages" ref="scrollEl">
      <ChatMessage v-for="(m, idx) in messages" :key="idx" :role="m.role" :content="m.content" />

      <div v-if="isLoading" class="status">Thinking…</div>
      <div v-if="error" class="error">{{ error }}</div>
    </div>

    <form class="composer" @submit.prevent="onSend">
      <input
        v-model="draft"
        class="input"
        type="text"
        placeholder="Type a message…"
        :disabled="isLoading"
        @keydown.enter.exact.prevent="onSend"
      />
      <button class="send" type="submit" :disabled="isLoading || !draft.trim()">
        Send
      </button>
    </form>
  </section>
</template>

<script setup>
import { onMounted, nextTick, ref, watch } from "vue";
import ChatMessage from "./ChatMessage.vue";
import { sendChatMessage } from "../services/chatApi.js";

const scrollEl = ref(null);
const draft = ref("");
const isLoading = ref(false);
const error = ref("");

const sessionId = getOrCreateSessionId();

const messages = ref(loadHistory() || [
  { role: "assistant", content: "Hi! Ask me an addition/subtraction question." }
]);

watch(
  messages,
  () => {
    localStorage.setItem("mcp_chat_history", JSON.stringify(messages.value));
    scrollToBottom();
  },
  { deep: true }
);

onMounted(() => {
  scrollToBottom();
});

async function onSend() {
  error.value = "";
  const text = draft.value.trim();
  if (!text || isLoading.value) return;

  messages.value.push({ role: "user", content: text });
  draft.value = "";
  isLoading.value = true;

  try {
    const data = await sendChatMessage({ sessionId, message: text });
    // const reply = data?.reply ?? "(no reply field returned)";
    // messages.value.push({ role: "assistant", content: reply });
    const reply = data?.reply ?? "(no reply field returned)";
    const tools = data?.tools_used?.length ? `\n\n[tools_used: ${data.tools_used.join(", ")}]` : "";
    messages.value.push({ role: "assistant", content: reply + tools });
  } catch (e) {
    error.value = String(e?.message || e);
  } finally {
    isLoading.value = false;
  }
}

function loadHistory() {
  try {
    const raw = localStorage.getItem("mcp_chat_history");
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function getOrCreateSessionId() {
  const key = "mcp_chat_session_id";
  let id = localStorage.getItem(key);
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem(key, id);
  }
  return id;
}

async function scrollToBottom() {
  await nextTick();
  const el = scrollEl.value;
  if (!el) return;
  el.scrollTop = el.scrollHeight;
}
</script>

<style scoped>
.card {
  width: min(920px, 100%);
  height: min(78vh, 860px);
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 18px;
  display: grid;
  grid-template-rows: 1fr auto;
  overflow: hidden;
  box-shadow: 0 10px 25px rgba(0,0,0,0.06);
}

.messages {
  padding: 16px;
  overflow: auto;
  background: #f9fafb;
}

.composer {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 10px;
  padding: 12px;
  border-top: 1px solid #e5e7eb;
  background: white;
}

.input {
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 10px 12px;
  font-size: 14px;
  outline: none;
}
.input:focus { border-color: #9ca3af; }

.send {
  border: 1px solid #111827;
  background: #111827;
  color: white;
  border-radius: 12px;
  padding: 10px 14px;
  cursor: pointer;
}
.send:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.status {
  margin-top: 8px;
  font-size: 12px;
  color: #6b7280;
}

.error {
  margin-top: 8px;
  font-size: 12px;
  color: #b91c1c;
}
</style>
