// Legacy file - template now uses inline <script> in index.html
// This file is kept for reference only
async function sendMessage() {
    const input = document.getElementById("question");
    const chatBox = document.getElementById("chat-box");
    const userText = input.value;
    if (!userText) return;
    chatBox.innerHTML += `<div class="message user">${userText}</div>`;
    const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userText })
    });
    const data = await response.json();
    chatBox.innerHTML += `<div class="message bot">${data.reply}</div>`;
    input.value = "";
    chatBox.scrollTop = chatBox.scrollHeight;
}