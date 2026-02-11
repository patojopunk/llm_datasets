<template>
  <div class="page">
    <header class="header">
      <div class="brand">
        <div class="logo">M</div>
        <div>
          <div class="title">MCP Chat</div>
          <div class="subtitle">Vue frontend → your backend → MCP tools</div>
        </div>
      </div>
      <button class="ghost" @click="clearChat">Clear</button>
    </header>

    <main class="main">
      <ChatWindow :key="resetKey" />
    </main>

    <footer class="footer">
      <span>Tip: ask “what is 12 + 34?” and the agent should call the MCP calculator.</span>
    </footer>
  </div>
</template>

<script setup>
import { ref } from "vue";
import ChatWindow from "./components/ChatWindow.vue";

const resetKey = ref(0);

function clearChat() {
  localStorage.removeItem("mcp_chat_history");
  resetKey.value++;
}
</script>

<style scoped>
.page {
  display: grid;
  grid-template-rows: auto 1fr auto;
  min-height: 100vh;
}

.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 18px;
  background: white;
  border-bottom: 1px solid #e5e7eb;
}

.brand { display: flex; gap: 12px; align-items: center; }
.logo {
  width: 36px; height: 36px;
  border-radius: 10px;
  display: grid; place-items: center;
  background: #111827;
  color: white;
  font-weight: 700;
}
.title { font-weight: 700; }
.subtitle { font-size: 12px; color: #6b7280; }

.ghost {
  border: 1px solid #e5e7eb;
  background: white;
  padding: 8px 10px;
  border-radius: 10px;
  cursor: pointer;
}
.ghost:hover { background: #f9fafb; }

.main {
  padding: 18px;
  display: grid;
  place-items: center;
}

.footer {
  padding: 10px 18px;
  color: #6b7280;
  font-size: 12px;
}
</style>
