# Predicting Emoji Demo

I used Open Ai's CLIP model on both text (tweets) and images of emoji's! <br/>
When the user provides a sentence, the CLIP model will suggest 4 from the following emoji's:
<br/>❤️ 😍 😂 💕 🔥 😊 😎 ✨ 💙 😘 📷 🇺🇸 ☀ 💜 😉 💯 😁 🎄 📸 😜 ☹️ 😭 😔 😡 💢 😤 😳 🙃 😩 😠 🙈 🙄

- Please install Python 3.10: https://www.python.org/downloads/release/python-3100/
- How to run <pre><code>pip install gradio torch more_itertools transformers
  python app.py
  </code></pre>
- Visit <pre><code>https://localhost:7680</code></pre>

## Note

I used the <b>Gradio package</b> to build this demo. <br/>
We can build the separate frontend for UI and make it call this demo as a backend api in the real product. <br/>
We can increase the choice of emojis by adding them in the emojis folder.😊
