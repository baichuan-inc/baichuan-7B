import json
import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from peft import PeftModel

st.set_page_config(page_title="Baichuan")
st.title("Baichuan")

model_path = "./baichuan-inc/Baichuan-7B"
# lora_path = './baichuan-inc/baichuan2-13b-iepile-lora'

@st.cache_resource
def init_model():
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.generation_config = GenerationConfig.from_pretrained(
        model_path
    )
    # model = PeftModel.from_pretrained(
    #     model,
    #     lora_path,
    # )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )
    return model, tokenizer


def clear_chat_history():
    del st.session_state.messages


def init_chat_history():
    with st.chat_message("assistant", avatar='🤖'):
        st.markdown("您好，我是百川大模型，很高兴为您服务🥰")

    if "messages" in st.session_state:
        for message in st.session_state.messages:
            avatar = '🧑‍💻' if message["role"] == "user" else '🤖'
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    else:
        st.session_state.messages = []

    return st.session_state.messages


def main():
    model, tokenizer = init_model()
    messages = init_chat_history()

    if prompt := st.chat_input("Shift + Enter 换行, Enter 发送"):
        with st.chat_message("user", avatar='🧑‍💻'):
            st.markdown(prompt)
        messages.append({"role": "user", "content": prompt})
        print(f"[user] {prompt}", flush=True)

        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=64, repetition_penalty=1.1)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        with st.chat_message("assistant", avatar='🤖'):
            st.markdown(response)
        messages.append({"role": "assistant", "content": response})

        print(json.dumps(messages, ensure_ascii=False), flush=True)

        st.button("清空对话", on_click=clear_chat_history)


if __name__ == "__main__":
    main()
