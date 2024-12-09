# -*- coding: utf-8 -*-

import streamlit as st
st.set_page_config(
    page_title='비상교육 출판이론교육 AI 교육용',
    layout='wide',
    page_icon=':robot_face:',
)

import pandas as pd
from openai import OpenAI
import pickle, zipfile
import base64
import io, os, re
from scipy import spatial
from PIL import Image

st.markdown("""<style>
    html, body, .stTextArea textarea {
        font-size: 14px;
    }
    section[data-testid="stSidebar"] {
        width: 200px !important;
        padding-top: 1rem;
    }
    div.row-widget.stRadio > div{flex-direction:row;}
</style>""", unsafe_allow_html=True)

def squeeze_spaces(s):
    s_without_spaces = re.sub(r'\s+', '', s)
    return s_without_spaces

def load_embeddings(pkz="embeddings.pkz"):
    with open(pkz, 'rb') as f:
        zip_data = io.BytesIO(f.read())

    with zipfile.ZipFile(zip_data, 'r') as zf:
        first_file_name = zf.namelist()[0]
        with zf.open(first_file_name) as pkl_file:
            pickle_data = io.BytesIO(pkl_file.read())
            data = pickle.load(pickle_data)
    return data

# Load standard questions from quests.txt
def load_standard_questions(file_path="quests.txt"):
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='cp949') as f:
                questions = f.readlines()
        except:
            with open(file_path, 'r', encoding='utf-8') as f:
                questions = f.readlines()        
        questions = [q.strip() for q in questions if q.strip()]
    else:
        questions = []
    return questions
    
standard_questions = load_standard_questions()

# Set model and map model names to OpenAI model IDs
EMBEDDING_MODEL = "text-embedding-3-large"
MAX_RETRIES = 2
chat_state = st.session_state

logo_image = "logo_image.jpg"

# Initialize
def init_chat(state):
    def set_chat(key, value):
        if key not in state:
            state[key] = value
    set_chat('prompt', [])
    set_chat('generated', [])
    set_chat('import_index', 0)
    if 'openai_client' not in state:
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            openai_key = st.sidebar.text_input("OpenAI API key", value="", type="password")
        if openai_key:
            state['openai_client'] = OpenAI(api_key=openai_key)
    st.cache_data.clear()

def clear_chat(state):
    def clear_one(key):
        del state[key][:]
    clear_one('prompt')
    clear_one('generated')

def compute_embedding(segment, model=EMBEDDING_MODEL):
    return st.session_state['openai_client'].embeddings.create(input=[segment], model=model).data[0].embedding

def related_strings(query, embeddings, relatedness_fn, top_n, threshold=0.28):
    query_embedding = compute_embedding(query)
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in embeddings.iterrows()
    ]
    strings_and_relatednesses = [
        (text, relatedness) 
        for text, relatedness in strings_and_relatednesses if relatedness > threshold 
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    
    if not strings_and_relatednesses:
        return [], []  # 결과가 없을 경우 빈 리스트 반환
    strings, relatednesses = zip(*strings_and_relatednesses[:top_n])
    return list(strings), list(relatednesses)

def query_message(query, prevs, model, embeddings):
    """Return a message for GPT, with relevant source texts pulled from embeddings."""

    strings = []
    relatednesses = []
    clues = ""
    
    strings, relatednesses = related_strings(
        query + prevs,
        embeddings,
        lambda subj1, subj2: 1 - spatial.distance.cosine(subj1, subj2),
        top_n
    )
    
    for i, string in enumerate(strings):  # TODO: Must check if the clues are within the token-budget
        clues += f"[Clue #{i+1}>>>]:\n{string.strip()}\n[<<<Clue #{i+1}]\n\n"

    unknown_ = "그 질문에 대한 정보가 부족하여 답을 알 수 없습니다."
    if not clues:
        return f"단서가 없으면 '{unknown_}' 라고 말해주세요.", [], []
    return f"""

다음 단서들을 참조하여 {tone}해주세요. 단서가 부족하면 '{unknown_}' 라고 답하고 없는 사실을 지어내지는 마세요. 
답변은 {expertise} 전문가의 용어나 문체를 적극 활용하고, 공손한 말투를 사용해주세요. 
활용된 단서에 포함된 ![Image](파일경로) 형식의 이미지 관련 정보는 최대한 응답 초반 혹은 중반에 나오도록 포함시켜주세요.

단서시작>>>>
{clues}
<<<<단서종료
""", strings, relatednesses
    
def img_to_bytes(img_path, scale=275):
    with Image.open(img_path) as img:
        original_width, original_height = img.size
        if original_width >= original_height:
            new_height = int((scale / original_width) * original_height)
            img_resized = img.resize((scale, new_height))
        else:
            new_width = int((scale / original_height) * original_width)
            img_resized = img.resize((new_width, scale))
        img_bytes = io.BytesIO()
        img_resized.save(img_bytes, format=img.format)
        encoded = base64.b64encode(img_bytes.getvalue()).decode()
    return encoded

def load_base64_image(base64_path):
    with open(base64_path, "r") as file:
        return file.read()

def convert_url_img_tags(text):
    # URL을 사용하는 Markdown 이미지 문법 패턴 탐색
    pattern = r"!\[(.*?)\]\((https?://[^\s\)]+)\)"
    
    def replace_with_img_tag(match):
        try:
            alt_text = match.group(1)
            img_url = match.group(2)
            # 이미지에 배경색과 여백, 크기 설정
            return f'''<p style="background-color: #f0f0f0; width: 100%; text-align: center; padding: 20px;"><img src="{img_url}" alt="{alt_text}" style="width: 50%; display: inline-block;"/></p>'''
        except Exception as e:
            # print(f"Error: {e}")
            return ""

    # 패턴을 찾아 <img> 태그로 변환
    return re.sub(pattern, replace_with_img_tag, text)

def convert_base64_paths_to_img_tags(text):
    # Base64 파일 경로 패턴 탐색 (예: resources/image_5_image40.txt)
    pattern = r"!\[\w+\]\((resources/[^\s]+\.txt)\)"
    
    def replace_with_img_tag(match):
        try:
            base64_path = match.group(1)
            img_base64 = load_base64_image(base64_path)
            # 이미지에 배경색과 여백, 크기 설정
            return f'''<p style="background-color: #f0f0f0; width: 100%; text-align: center; padding: 20px;"><img src="data:image/png;base64,{img_base64}" alt="Image" style="width: 50%; display: inline-block;"/></p>'''
        except:
            return ""

    # 패턴을 찾아 <img> 태그로 변환
    return re.sub(pattern, replace_with_img_tag, text)

def interact():
    st.markdown(f"<h2 style='text-align: center;'><font color='green'>{subject}</font></h2>",
        unsafe_allow_html=True)

    if intro:
        st.markdown("")
        st.markdown(f"<p style='text-align: center;'>{intro}</p>", unsafe_allow_html=True)

    st.markdown("")

    def img_to_html(img_path):
        img_html = f"<img src='data:image/png;base64,{img_to_bytes(img_path)}' style='display: block; margin-left: auto; margin-right: auto;'>"
        return img_html

    for _ in range(2):
        st.sidebar.markdown("")
        
    if logo_image and os.path.exists(logo_image):
        image_html = img_to_html(logo_image)
        st.sidebar.markdown("<p style='text-align: center;'>"+image_html+"</p>", unsafe_allow_html=True)
        st.sidebar.markdown("")

    init_chat(chat_state)
    
    # Embeddings
    if 'embeddings' not in chat_state:
        with st.spinner("색인 정보를 로드하는 중..."):
            chat_state['embeddings'] = load_embeddings()
    
    # Generate a response
    def generate_response(query, show_clues=False):
        prevs = ""
        # if len(chat_state.prompt) > 0:
        #     for i, _prompt in enumerate(chat_state.prompt):
        #         prevs += f"({_prompt}) "
        context = ""
        chat_state.lookup = None
        with st.spinner("단서 탐색 중..."):
            context, clues, scores = query_message(query, prevs, model, chat_state.embeddings)
            chat_state.lookup = (clues, scores)
                
        if os.path.exists('addendum.txt'):
            try:
                with open('addendum.txt', 'r', encoding='cp949') as file:
                    content = file.read()
            except:
                with open('addendum.txt', 'r', encoding='utf-8') as file:
                    content = file.read()
            context += content
   
        # if prevs:
        #      context += f"\n\n이전에 {prevs.strip()}과 같은 질문들을 한 적이 있으니 아래 질문의 맥락을 파악하기 위해 참고만 하고 그 질문들에 대한 단서를 참조하거나 직접적인 답은 절대 하지 마세요.\n"
    
        summary = ""
        if show_clues:
            summary = "\n\n단서를 참조하여 답변을 하는 경우 2건 내외의 핵심 단서들을 (내용 / 출처출처) 형태로 간략하게 답변 뒤에 요약해주세요."
        context += f"\n\n[!!!! 이 질문에 답해주세요: ({query}) !!!!{summary}]"
            
        user_content = f"{context}\n\n기본적으로 {language} 언어로 {expertise} 전문가인척 응답하되, {tone}하세요."
        user_content += "또한 활용된 단서에 이미지가 있으면 응답에도 그대로 포함시켜주세요. 영어가 아닌 경우 콤마의 사용을 최소화해주세요."
        user_content = re.sub(r'\n\s*\n', '\n\n', user_content)
        print(user_content)
    
        system_content = f"당신은 {tone}하는 {expertise} {language} 원어민 전문가입니다."
        
        if True:
        # with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            stream = chat_state['openai_client'].chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ],
                n=1,
                # top_p=1,
                stop=None,
                stream=True)
            for chunk in stream:
                cpart = chunk.choices[0].delta.content
                if cpart is not None:
                    full_response += cpart.replace('~', '-')
                message_placeholder.markdown(full_response + "_", unsafe_allow_html=True)
            new_answer = full_response.strip()
            message_placeholder.markdown(new_answer, unsafe_allow_html=True)
    
        return new_answer

    def message_response(text):
        # st.success(text)
        st.markdown(f"<div style='background-color: #efefef; color: black; padding: 15px; font-size: 13px;'>{text}</div><br>", unsafe_allow_html=True)

    for i in range(len(chat_state.generated)):
        # st.chat_message("user").write(chat_state.prompt[i])
        # st.chat_message("assistant").write(chat_state.generated[i])
        st.info(chat_state.prompt[i])
        st.markdown(chat_state.generated[i], unsafe_allow_html=True)
        st.markdown("")
        
    user_input = ""
    selected_question = ""
    if standard_questions:
        with st.sidebar.expander("질문선택"):
            selected_question = st.selectbox("질문을 선택하세요", 
                standard_questions, 
                index=None, 
                key=f"user_select",
                label_visibility="collapsed")
            if selected_question and selected_question != "":
                user_input = selected_question

    text_input = st.chat_input("무엇을 도와드릴까요?", key="user_input")
    if text_input:
        user_input = text_input

    if user_input:
        # st.chat_message("user").write(user_input)
        st.info(user_input)
        
        retries = 1
        while retries <= MAX_RETRIES:
            try:
                generated = generate_response(user_input, show_clues)
                break
            except Exception as e:
                error_msgs = str(e)
                if "reduce the length of the messages" in error_msgs:
                    st.error(error_msgs)
                    break
                else:
                    retries = MAX_RETRIES + 2
                    break
        # After "while"...
        chat_state['prompt'].append(user_input.strip())
        if retries <= MAX_RETRIES:
            print("\n\n", generated)
            generated = convert_url_img_tags(convert_base64_paths_to_img_tags(generated.strip()))
            chat_state['generated'].append(generated)  
            # st.rerun()
        else:
            chat_state['generated'].append(error_msgs)
            if retries == MAX_RETRIES + 1:
                st.error("잠시 후에 다시 시도해 주세요.")
                
    if selected_question:
        del chat_state['user_select']
        chat_state.user_select = None
        st.rerun()
    
    if user_input:
        st.rerun()

    if len(chat_state.prompt) > 0:
        cols = st.columns((19, 1))
        with cols[1]:
            if st.button(":wastebasket:", help=':red[마지막 대화 삭제]', key="ok_chat_delete"):
                chat_state.prompt.pop()
                chat_state.generated.pop()
                st.rerun()

    # Retrievals
    if chat_state.get('lookup', None):
        with st.sidebar.expander("자료출처"):
            st.dataframe(pd.DataFrame({
                "텍스트": chat_state.lookup[0],
                "연관성": chat_state.lookup[1]
            }), hide_index=True)
    
    # Bells and whistles
    with st.sidebar.expander("내보내기", expanded=False):
        def to_csv(dataframe):
            csv_buffer = io.StringIO()
            dataframe.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            return io.BytesIO(csv_buffer.getvalue().encode("cp949"))
    
        def to_html(dataframe):
            html = dataframe.to_html(index=False, escape=False).replace("\\n", '\0')
            return '<meta charset="utf-8">\n' + html

        with st.container():
            file_type = st.radio(label="Format",
                options=("As CSV", "As HTML"), label_visibility="collapsed")
            file_type = file_type.split()[-1].lower()
            def build_data(chat):
                return (to_csv if file_type == "csv" else to_html)(chat)
            file_name = st.text_input("파일명", squeeze_spaces(subject))
            if file_name:
                file_path = f"{file_name}.{file_type}"
                download = st.download_button(
                    label="확인",
                    data=build_data(pd.DataFrame({
                        'Prompt': chat_state['prompt'],
                        'Response': chat_state['generated'],
                    })),
                    file_name=file_path,
                    mime=f'text/{file_type}')

    with st.sidebar.expander("불러오기", expanded=False):
        conversation = st.file_uploader('대화 파일 업로드', key=f'import_file_{chat_state.import_index}', label_visibility='collapsed')
        if conversation and st.button("확인",
                key="ok_restore", help="이 메뉴를 실행하면 현재 진행중인 대화가 지워집니다!"):
            # Read the bytes of the file into a bytes object
            try:
                saved_chat = pd.read_csv(conversation, encoding="cp949")
            except:
                saved_chat = pd.read_csv(conversation, encoding="utf-8")
            clear_chat(chat_state)
            chat_state['prompt'] = saved_chat.Prompt.tolist()
            chat_state['generated'] = saved_chat.Response.tolist()
            chat_state['import_index'] += 1
            st.rerun()

###

model = 'gpt-4o-mini'
top_n = 3

expertise = '비상교육'
tone = '사실에 기반하여 응답'
temperature = 0.5
language = 'Korean'

subject = '비상교육 출판이론교육 AI 교육용'
intro = '* 제공된 정보가 정확하지 않을 수 있으니, 이 정보를 참고자료로만 사용하고 필요하면 직접 더 확인해 보세요.'
show_clues = False

###
# Launch the bot
interact()
