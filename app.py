import os
import gradio as gr

from answer import answer_question

def format_context(context):
    result = "<h2 style='color: #ff7800;'>Relevant Context</h2>\n\n"

    for document in context:
        source = document.metadata.get("source", "Unknown source")
        result += (
            f"<span style='color: #ff7800;'>Source: {source}</span>\n\n"
        )
        result += document.page_content + "\n\n"

    return result


def chat(history):
    latest_question = history[-1][0]

    prior_history = [
        {
            "role": "user",
            "content": message[0],
        }
        for message in history[:-1]
        if message[0]
    ]

    answer, context = answer_question(
        latest_question,
        prior_history,
    )

    history[-1][1] = answer

    return history, format_context(context)


def main():
    def put_message_in_chatbot(message, history):
        return "", history + [[message, None]]

    theme = gr.themes.Soft(
        font=["Inter", "system-ui", "sans-serif"]
    )

    with gr.Blocks(
        title="My Runbook Assistant",
        theme=theme,
    ) as ui:
        gr.Markdown(
            "# My Runbook Assistant\n"
            "Ask me anything about live production issues!"
        )

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=600,
                    type="tuples",
                    show_copy_button=True,
                )

                message = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything about yourself...",
                    show_label=False,
                )

            with gr.Column(scale=1):
                context_markdown = gr.Markdown(
                    label="Retrieved Context",
                    value="*Retrieved context will appear here*",
                    height=600,
                )

        message.submit(
            put_message_in_chatbot,
            inputs=[message, chatbot],
            outputs=[message, chatbot],
            api_name=False,
        ).then(
            chat,
            inputs=chatbot,
            outputs=[chatbot, context_markdown],
            api_name=False,
        )
    # Grab the port Render gives you, or fall back to 7860 locally
    port = int(os.environ.get("PORT", 7860))

    ui.launch(
        server_name="0.0.0.0", # allows Render to detect the app
        server_port=port,
        share=False,
        show_error=True,
        show_api=False
    )

if __name__ == "__main__":
    main()