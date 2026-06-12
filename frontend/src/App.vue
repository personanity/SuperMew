<template>
  <div class="app-wrapper">
    <!-- Sidebar (Left) -->
    <Sidebar />

    <!-- Main Content Area -->
    <main class="main-content">
      <!-- If not authenticated, show AuthPanel -->
      <AuthPanel v-if="!authStore.isAuthenticated" />

      <template v-else>
        <!-- Document Management Settings View -->
        <DocumentSettings v-if="chatStore.activeNav === 'settings'" />

        <!-- History Sidebar (Slides in on right if enabled) -->
        <HistorySidebar />

        <!-- Chat Area (Visible when settings is not active) -->
        <ChatArea v-show="chatStore.activeNav !== 'settings'" />
      </template>
    </main>
  </div>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted } from 'vue';
import Sidebar from '@/components/Sidebar.vue';
import AuthPanel from '@/components/AuthPanel.vue';
import HistorySidebar from '@/components/HistorySidebar.vue';
import ChatArea from '@/components/Chat/ChatArea.vue';
import DocumentSettings from '@/components/Documents/DocumentSettings.vue';

import { useAuthStore } from '@/stores/auth';
import { useChatStore } from '@/stores/chat';

const authStore = useAuthStore();
const chatStore = useChatStore();

const handleUnauthorized = () => {
  authStore.handleLogout();
  alert('登录已过期，请重新登录');
};

onMounted(async () => {
  window.addEventListener('unauthorized', handleUnauthorized);
  
  if (authStore.token) {
    try {
      await authStore.fetchMe();
    } catch (_) {
      authStore.handleLogout();
    }
  }
});

onUnmounted(() => {
  window.removeEventListener('unauthorized', handleUnauthorized);
});
</script>
