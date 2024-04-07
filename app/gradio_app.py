import gradio as gr
from langchain.tools import Tool
from langchain_community.utilities import GoogleSearchAPIWrapper
import os 

from langchain.tools import Tool
from langchain_community.utilities import GoogleSearchAPIWrapper


def get_search(query:str="", k:int=1): # get the top-k resources with google
    search = GoogleSearchAPIWrapper(k=k)
    def search_results(query):
        return search.results(query, k)
    tool = Tool(
        name="Google Search Snippets",
        description="Search Google for recent results.",
        func=search_results,
    )
    ref_text = tool.run(query)
    if 'Result' not in ref_text[0].keys():
        return ref_text
    else:
        return None

from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.document_loaders import AsyncHtmlLoader
def get_page_content(link:str):
    loader = AsyncHtmlLoader([link])
    docs = loader.load()
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)
    if len(docs_transformed) > 0:
        return docs_transformed[0].page_content
    else:
        return None

import tiktoken
def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def chunk_text_by_sentence(text, chunk_size=2048):
    """Chunk the $text into sentences with less than 2k tokens."""
    sentences = text.split('. ')
    chunked_text = []
    curr_chunk = []
    # 逐句添加文本片段，确保每个段落都小于2k个token
    for sentence in sentences:
        if num_tokens_from_string(". ".join(curr_chunk)) + num_tokens_from_string(sentence) + 2 <= chunk_size:
            curr_chunk.append(sentence)
        else:
            chunked_text.append(". ".join(curr_chunk))
            curr_chunk = [sentence]
    # 添加最后一个片段
    if curr_chunk:
        chunked_text.append(". ".join(curr_chunk))
    return chunked_text[0]

def chunk_text_front(text, chunk_size = 2048):
    '''
    get the first `trunk_size` token of text
    '''
    chunked_text = ""
    tokens = num_tokens_from_string(text)
    if tokens < chunk_size:
        return text
    else:
        ratio = float(chunk_size) / tokens
        char_num = int(len(text) * ratio)
        return text[:char_num]

def chunk_texts(text, chunk_size = 2048):
    '''
    trunk the text into n parts, return a list of text
    [text, text, text]
    '''
    tokens = num_tokens_from_string(text)
    if tokens < chunk_size:
        return [text]
    else:
        texts = []
        n = int(tokens/chunk_size) + 1
        # 计算每个部分的长度
        part_length = len(text) // n
        # 如果不能整除，则最后一个部分会包含额外的字符
        extra = len(text) % n
        parts = []
        start = 0

        for i in range(n):
            # 对于前extra个部分，每个部分多分配一个字符
            end = start + part_length + (1 if i < extra else 0)
            parts.append(text[start:end])
            start = end
        return parts
    
from datetime import datetime

from openai import OpenAI
import openai
import os

chatgpt_system_prompt = f'''
You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.
Knowledge cutoff: 2023-04
Current date: {datetime.now().strftime('%Y-%m-%d')}
'''

def get_draft(question):
    # Getting the draft answer
    draft_prompt = '''
IMPORTANT:
Try to answer this question/instruction with step-by-step thoughts and make the answer more structural.
Use `\n\n` to split the answer into several paragraphs.
Just respond to the instruction directly. DO NOT add additional explanations or introducement in the answer unless you are asked to.
'''
    # openai_client = OpenAI(api_key=openai.api_key)
    openai_client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))
    draft = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": chatgpt_system_prompt
            },
            {
                "role": "user",
                "content": f"{question}" + draft_prompt
            }
        ],
        temperature = 1.0
    ).choices[0].message.content
    return draft

def split_draft(draft, split_char = '\n\n'):
    # 将draft切分为多个段落
    # split_char: '\n\n'
    paragraphs = draft.split(split_char)
    draft_paragraphs = [para for para in paragraphs if len(para)>5]
    # print(f"The draft answer has {len(draft_paragraphs)}")
    return draft_paragraphs

def split_draft_openai(question, answer, NUM_PARAGRAPHS = 4):
    split_prompt = f'''
Split the answer of the question into multiple paragraphs with each paragraph containing a complete thought.
The answer should be splited into less than {NUM_PARAGRAPHS} paragraphs.
Use ## as splitting char to seperate the paragraphs.
So you should output the answer with ## to split the paragraphs.
**IMPORTANT**
Just output the query directly. DO NOT add additional explanations or introducement in the answer unless you are asked to.
'''
    openai_client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))
    splited_answer = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": chatgpt_system_prompt
            },
            {
                "role": "user",
                "content": f"##Question: {question}\n\n##Response: {answer}\n\n##Instruction: {split_prompt}"
            }
        ],
        temperature = 1.0
    ).choices[0].message.content
    split_draft_paragraphs = split_draft(splited_answer, split_char = '##')
    return split_draft_paragraphs

def get_query(question, answer):
    query_prompt = '''
I want to verify the content correctness of the given question, especially the last sentences.
Please summarize the content with the corresponding question.
This summarization will be used as a query to search with Bing search engine.
The query should be short but need to be specific to promise Bing can find related knowledge or pages.
You can also use search syntax to make the query short and clear enough for the search engine to find relevant language data.
Try to make the query as relevant as possible to the last few sentences in the content.
**IMPORTANT**
Just output the query directly. DO NOT add additional explanations or introducement in the answer unless you are asked to.
'''
    # openai_client = OpenAI(api_key = openai.api_key)
    openai_client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))
    query = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": chatgpt_system_prompt
            },
            {
                "role": "user",
                "content": f"##Question: {question}\n\n##Content: {answer}\n\n##Instruction: {query_prompt}"
            }
        ],
        temperature = 1.0
    ).choices[0].message.content
    return query

def get_content(query):
    res = get_search(query, 1)
    if not res:
        print(">>> No good Google Search Result was found")
        return None
    search_results = res[0]
    link = search_results['link'] # title, snippet
    res = get_page_content(link)
    if not res:
        print(f">>> No content was found in {link}")
        return None
    retrieved_text = res
    trunked_texts = chunk_texts(retrieved_text, 1500)
    trunked_texts = [trunked_text.replace('\n', " ") for trunked_text in trunked_texts]
    return trunked_texts

def get_revise_answer(question, answer, content):
    revise_prompt = '''
I want to revise the answer according to retrieved related text of the question in WIKI pages.
You need to check whether the answer is correct.
If you find some errors in the answer, revise the answer to make it better.
If you find some necessary details are ignored, add it to make the answer more plausible according to the related text.
If you find the answer is right and do not need to add more details, just output the original answer directly.
**IMPORTANT**
Try to keep the structure (multiple paragraphs with its subtitles) in the revised answer and make it more structual for understanding.
Add more details from retrieved text to the answer.
Split the paragraphs with \n\n characters.
Just output the revised answer directly. DO NOT add additional explanations or annoucement in the revised answer unless you are asked to.
'''
    # openai_client = OpenAI(api_key = openai.api_key)
    openai_client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))
    revised_answer = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
                {
                    "role": "system",
                    "content": chatgpt_system_prompt
                },
                {
                    "role": "user",
                    "content": f"##Existing Text in Wiki Web: {content}\n\n##Question: {question}\n\n##Answer: {answer}\n\n##Instruction: {revise_prompt}"
                }
            ],
            temperature = 1.0
    ).choices[0].message.content
    return revised_answer

def get_reflect_answer(question, answer):
    reflect_prompt = '''
Give a title for the answer of the question.
And add a subtitle to each paragraph in the answer and output the final answer using markdown format. 
This will make the answer to this question look more structured for better understanding.
**IMPORTANT**
Try to keep the structure (multiple paragraphs with its subtitles) in the response and make it more structual for understanding.
Split the paragraphs with \n\n characters.
Just output the revised answer directly. DO NOT add additional explanations or annoucement in the revised answer unless you are asked to.
'''
    openai_client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))
    reflected_answer = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
                {
                    "role": "system",
                    "content": chatgpt_system_prompt
                },
                {
                    "role": "user",
                    "content": f"##Question:\n{question}\n\n##Answer:\n{answer}\n\n##Instruction:\n{reflect_prompt}"
                }
            ],
            temperature = 1.0
    ).choices[0].message.content
    return reflected_answer

def get_query_wrapper(q, question, answer):
    result = get_query(question, answer)
    q.put(result)  # 将结果放入队列

def get_content_wrapper(q, query):
    result = get_content(query)
    q.put(result)  # 将结果放入队列

def get_revise_answer_wrapper(q, question, answer, content):
    result = get_revise_answer(question, answer, content)
    q.put(result)

def get_reflect_answer_wrapper(q, question, answer):
    result = get_reflect_answer(question, answer)
    q.put(result)

from multiprocessing import Process, Queue
def run_with_timeout(func, timeout, *args, **kwargs):
    q = Queue()  # 创建一个Queue对象用于进程间通信
    # 创建一个进程来执行传入的函数，将Queue和其他*args、**kwargs作为参数传递
    p = Process(target=func, args=(q, *args), kwargs=kwargs)
    p.start()
    # 等待进程完成或超时
    p.join(timeout)
    if p.is_alive():
        print(f"{datetime.now()} [INFO] Function {str(func)} running timeout ({timeout}s), terminating...")
        p.terminate()  # 终止进程
        p.join()  # 确保进程已经终止
        result = None  # 超时情况下，我们没有结果
    else:
        print(f"{datetime.now()} [INFO] Function {str(func)} executed successfully.")
        result = q.get()  # 从队列中获取结果
    return result

from difflib import unified_diff
from IPython.display import display, HTML

def generate_diff_html(text1, text2):
    diff = unified_diff(text1.splitlines(keepends=True),
                        text2.splitlines(keepends=True),
                        fromfile='text1', tofile='text2')

    diff_html = ""
    for line in diff:
        if line.startswith('+'):
            diff_html += f"<div style='color:green;'>{line.rstrip()}</div>"
        elif line.startswith('-'):
            diff_html += f"<div style='color:red;'>{line.rstrip()}</div>"
        elif line.startswith('@'):
            diff_html += f"<div style='color:blue;'>{line.rstrip()}</div>"
        else:
            diff_html += f"{line.rstrip()}<br>"
    return diff_html

newline_char = '\n'

def rat(question):
    print(f"{datetime.now()} [INFO] Generating draft...")
    draft = get_draft(question)
    print(f"{datetime.now()} [INFO] Return draft.")
    # print(f"##################### DRAFT #######################")
    # print(draft)
    # print(f"#####################  END  #######################")

    print(f"{datetime.now()} [INFO] Processing draft ...")
    # draft_paragraphs = split_draft(draft)
    draft_paragraphs = split_draft_openai(question, draft)
    print(f"{datetime.now()} [INFO] Draft is splitted into {len(draft_paragraphs)} sections.")
    answer = ""
    for i, p in enumerate(draft_paragraphs):
        # print(str(i)*80)
        print(f"{datetime.now()} [INFO] Revising {i+1}/{len(draft_paragraphs)} sections ...")
        answer = answer + '\n\n' + p
        # print(f"[{i}/{len(draft_paragraphs)}] Original Answer:\n{answer.replace(newline_char, ' ')}")

        # query = get_query(question, answer)
        print(f"{datetime.now()} [INFO] Generating query ...")
        res = run_with_timeout(get_query_wrapper, 30, question, answer)
        if not res:
            print(f"{datetime.now()} [INFO] Generating query timeout, skipping...")
            continue
        else:
            query = res
        print(f">>> {i}/{len(draft_paragraphs)} Query: {query.replace(newline_char, ' ')}")

        print(f"{datetime.now()} [INFO] Crawling network pages ...")
        # content = get_content(query)
        res = run_with_timeout(get_content_wrapper, 30, query)
        if not res:
            print(f"{datetime.now()} [INFO] Parsing network pages timeout, skipping ...")
            continue
        else:
            content = res

        LIMIT = 2
        for j, c in enumerate(content):
            if  j >= LIMIT: # limit rge number of network pages
                break
            print(f"{datetime.now()} [INFO] Revising answers with retrieved network pages...[{j}/{min(len(content),LIMIT)}]")
            # answer = get_revise_answer(question, answer, c)
            res = run_with_timeout(get_revise_answer_wrapper, 30, question, answer, c)
            if not res:
                print(f"{datetime.now()} [INFO] Revising answers timeout, skipping ...")
                continue
            else:
                diff_html = generate_diff_html(answer, res)
                display(HTML(diff_html))
                answer = res
            print(f"{datetime.now()} [INFO] Answer revised [{j}/{min(len(content),3)}]")
        # print(f"[{i}/{len(draft_paragraphs)}] REVISED ANSWER:\n {answer.replace(newline_char, ' ')}")
        # print()
    res = run_with_timeout(get_reflect_answer_wrapper, 30, question, answer)
    if not res:
        print(f"{datetime.now()} [INFO] Reflecting answers timeout, skipping next steps...")
    else:
        answer = res
    return draft, answer

page_title = "RAT: Retrieval Augmented Thoughts Elicit Context-Aware Reasoning in Long-Horizon Generation"
page_md = """
# RAT: Retrieval Augmented Thoughts Elicit Context-Aware Reasoning in Long-Horizon Generation

We explore how iterative revising a chain of thoughts with the help of information retrieval significantly improves large language models' reasoning and generation ability in long-horizon generation tasks, while hugely mitigating hallucination. In particular, the proposed method — retrieval-augmented thoughts (RAT) — revises each thought step one by one with retrieved information relevant to the task query, the current and the past thought steps, after the initial zero-shot CoT is generated.

Applying RAT to various base models substantially improves their performances on various long-horizon generation tasks; on average of relatively increasing rating scores by 13.63% on code generation, 16.96% on mathematical reasoning, 19.2% on creative writing, and 42.78% on embodied task planning.

Feel free to try our demo!

"""

def clear_func():
    return "", "", ""

def set_openai_api_key(api_key):
    if api_key and api_key.startswith("sk-") and len(api_key) > 50:
        os.environ["OPENAI_API_KEY"] = api_key

with gr.Blocks(title = page_title) as demo:
   
    gr.Markdown(page_md)

    with gr.Row():
        chatgpt_box = gr.Textbox(
            label = "ChatGPT",
            placeholder = "Response from ChatGPT with zero-shot chain-of-thought.",
            elem_id = "chatgpt"
        )

    with gr.Row():
        stream_box = gr.Textbox(
            label = "Streaming",
            placeholder = "Interactive response with RAT...",
            elem_id = "stream",
            lines = 10,
            visible = False
        )
    
    with gr.Row():
        rat_box = gr.Textbox(
            label = "RAT",
            placeholder = "Final response with RAT ...",
            elem_id = "rat",
            lines = 6
        )

    with gr.Column(elem_id="instruction_row"):
        with gr.Row():
            instruction_box = gr.Textbox(
                label = "instruction",
                placeholder = "Enter your instruction here",
                lines = 2,
                elem_id="instruction",
                interactive=True,
                visible=True
            )
        # with gr.Row():
        #     model_radio = gr.Radio(["gpt-3.5-turbo", "gpt-4", "GPT-4-turbo"], elem_id="model_radio", value="gpt-3.5-turbo", 
        #                         label='GPT model', 
        #                         show_label=True,
        #                         interactive=True, 
        #                         visible=True) 
        #     openai_api_key_textbox = gr.Textbox(
        #         label='OpenAI API key',
        #         placeholder="Paste your OpenAI API key (sk-...) and hit Enter", 
        #         show_label=True, 
        #         lines=1, 
        #         type='password')
            
    # openai_api_key_textbox.change(set_openai_api_key,
    #     inputs=[openai_api_key_textbox],
    #     outputs=[])

    with gr.Row():
        submit_btn = gr.Button(
            value="submit", visible=True, interactive=True
        )
        clear_btn = gr.Button(
            value="clear", visible=True, interactive=True
        )
        regenerate_btn = gr.Button(
            value="regenerate", visible=True, interactive=True
        )

    submit_btn.click(
        fn = rat,
        inputs = [instruction_box],
        outputs = [chatgpt_box, rat_box]
    )

    clear_btn.click(
        fn = clear_func,
        inputs = [],
        outputs = [instruction_box, chatgpt_box, rat_box]
    )

    regenerate_btn.click(
        fn = rat,
        inputs = [instruction_box],
        outputs = [chatgpt_box, rat_box]
    )

    examples = gr.Examples(
        examples=[
            # "I went to the supermarket yesterday.", 
            # "Helen is a good swimmer."
            "Write a survey of retrieval-augmented generation in Large Language Models.",
            "Introduce Jin-Yong's life and his works.",
            "Summarize the American Civil War according to the timeline.",
            "Describe the life and achievements of Marie Curie"
            ],
        inputs=[instruction_box]
        )

demo.launch(server_name="0.0.0.0", debug=True)