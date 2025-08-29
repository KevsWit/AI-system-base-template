import uuid
import gradio as gr
from .main import Maki
from .utils.logging import start_run

maki = Maki()

def qa_respond(msg, chat_history, sid):
    # fija session_id de logs para esta llamada
    start_run(session_id=sid)

    history = chat_history or []
    messages = []
    for item in history:
        if isinstance(item, dict) and "role" in item and "content" in item:
            messages.append(item)
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            u, a = item
            if isinstance(u, dict): messages.append(u)
            elif isinstance(u, str): messages.append({"role": "user", "content": u})
            if isinstance(a, dict): messages.append(a)
            elif isinstance(a, str): messages.append({"role": "assistant", "content": a})

    if msg and msg.strip():
        out = maki.ask(msg)
        answer = out["final_answer"]
        messages.extend([
            {"role": "user", "content": msg},
            {"role": "assistant", "content": answer},
        ])

    # outputs: textbox vacío, messages, y devolver el mismo sid
    return "", messages, sid

def build_demo():
    with gr.Blocks(title="Maki QA") as demo:
        sid = gr.State(str(uuid.uuid4()))
        gr.Markdown("## Maki – QA del análisis (demo)")
        with gr.Tab("QA del análisis"):
            qa_chat = gr.Chatbot(label="💬 Pregunta sobre el análisis", height=320, show_label=True, type="messages")
            with gr.Row():
                qa_msg = gr.Textbox(placeholder="Ej: ...", label="Pregunta", lines=2, scale=4)
                qa_btn = gr.Button("💬 Preguntar", variant="primary", scale=1)

            # 👇 EXACTAMENTE 3 inputs y 3 outputs
            qa_btn.click(qa_respond, inputs=[qa_msg, qa_chat, sid], outputs=[qa_msg, qa_chat, sid])
            qa_msg.submit(qa_respond, inputs=[qa_msg, qa_chat, sid], outputs=[qa_msg, qa_chat, sid])
    return demo

if __name__ == "__main__":
    build_demo().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, debug=True, show_error=True)
