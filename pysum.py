import MeCab
import os

os.system('git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git && cd mecab-ipadic-neologd && ./bin/install-mecab-ipadic-neologd -n -y -u -p $PWD')
os.system('git clone --depth 1 https://github.com/neologd/mecab-unidic-neologd.git && cd mecab-unidic-neologd && ./bin/install-mecab-unidic-neologd -n -y -u -p $PWD')

import pandas as pd
import base64
import streamlit as st
from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
from pysummarization.tokenizabledoc.mecab_tokenizer import MeCabTokenizer
from pysummarization.abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor
from pysummarization.nlp_base import NlpBase
from pysummarization.similarityfilter.tfidf_cosine import TfIdfCosine
from pysummarization.tokenizabledoc.simple_tokenizer import SimpleTokenizer

input_data = None
document = None

def preprocessing(input_data):
	document = "".join(input_data).replace("　", "").replace(" ", "")
	
	return document

def summarize(document):
	
	# 自動要約のオブジェクト
	auto_abstractor = AutoAbstractor()

	# 日本語のトークナイザー設定（MeCab使用）
	auto_abstractor.tokenizable_doc = MeCabTokenizer() #.set_mecab_system_dic("/usr/lib/aarch64-linux-gnu/mecab/dic/mecab-ipadic-neologd")

	# 文のリストを作成するための区切り文字を設定
	auto_abstractor.delimiter_list = ["。", "\n"]

	# ドキュメントを抽象化およびフィルタリングするオブジェクト
	abstractable_doc = TopNRankAbstractor()

	# 変数を渡し文書を要約
	result_dict1 = auto_abstractor.summarize(document, abstractable_doc)
 
	_= """result_dictは辞書型となっています。
	result_dict{
			"summarize_result": "要約された文のリスト。", 
			"scoring_data":     "スコアのリスト（重要度のランク）。"
	}
	"""
	
	# similarity_filter機能を利用した要約
	# NLPオブジェクト
	nlp_base = NlpBase()

	# トークナイザーを設定します。 これは、MeCabを使用した日本語のトークナイザーです
	nlp_base.tokenizable_doc = MeCabTokenizer() #.set_mecab_system_dic("/usr/lib/aarch64-linux-gnu/mecab/dic/mecab-ipadic-neologd")

	# 「類似性フィルター」のオブジェクト。 このオブジェクトによって観察される類似性は、Tf-Idfベクトルのいわゆるコサイン類似性です
	similarity_filter = TfIdfCosine()

	# NLPオブジェクト設定
	similarity_filter.nlp_base = nlp_base

	# 類似性limit：類似性がlimitを超える文は切り捨てられます。
	similarity_filter.similarity_limit = similarity_limit

	# 自動要約のオブジェクト
	auto_abstractor = AutoAbstractor()

	# 日本語のトークナイザー設定（MeCab使用）
	auto_abstractor.tokenizable_doc = MeCabTokenizer() #.set_mecab_system_dic("/usr/lib/aarch64-linux-gnu/mecab/dic/mecab-ipadic-neologd")

	# 文のリストを作成するための区切り文字を設定
	auto_abstractor.delimiter_list = ["。", "\n"]

	# ドキュメントを抽象化およびフィルタリングするオブジェクト
	abstractable_doc = TopNRankAbstractor()

	# 文書要約（similarity_filter機能追加）
	result_dict2 = auto_abstractor.summarize(document, abstractable_doc, similarity_filter)

	return result_dict1, result_dict2

	# for sentence in result_dict["summarize_result"]:
	# 	st.write(sentence)

# インターフェイスの表示
st.title('文章要約アプリ')
st.write("・Pythonのテキスト自動要約ライブラリ「pysummarization」を使用した「抽出型」の文書要約アプリです。")
st.write("・テキストファイルをアップロードするとを通常の要約と類似性フィルター（Similarity Filter）機能を使用した要約を返してくれます。")

# ファイル選択
st.write("### 要約ファイル選択")
st.write("※対応言語は日本語となります。")
uploaded_file = st.file_uploader("テキストファイルをアップロードしてください。", ["txt"])

if uploaded_file is not None:
	content = uploaded_file.read()
	input_data = content.decode()
	
if input_data is not None:
	st.write(input_data)

# 類似性フィルターカットオフ設定（Defalt = 0.25）
st.sidebar.write("### 類似性フィルター（Similarity Filter）カットオフ設定")
similarity_limit = st.sidebar.slider(
	"類似性フィルターカットオフの値を指定してください。指定した値以上の類似度の文は切り捨てられます。この値によって要約結果が変わります。", 0.05, 0.5, 0.25, 0.05)

# 要約実行関数を呼び出し
st.write("### 要約実行")

st.write("要約を開始しますか？")
if st.button("開始"):
	document = preprocessing(input_data)
	
# 要約を実行し結果の表示
if document is not None:
	st.write("### 結果の表示")
	result_dict1, result_dict2 = summarize(document)
	
 # データフレーム化
	doc0 = "".join(s for s in document)
	doc1 = result_dict1["summarize_result"]
	doc2 = result_dict2["summarize_result"]
	doc1 = "".join(s for s in doc1)
	doc2 = "".join(s for s in doc2)
	lst1 = ["原文書", "要約文書", "要約文書:SF使用"]
	lst2 = [doc0, doc1, doc2]
	df = pd.DataFrame(list(zip(lst1, lst2)), columns = ["Class", "Content"])
	df = df.replace("\n", "", regex=True)

	with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.colheader_justify','light', 'display.width', 2000, 'display.max_colwidth', 500):
			df = df.stack().str.lstrip().unstack()
			df_st = df.style.set_properties(**{'text-align': 'left'})

	st.table(df_st)
 
	csv = df.to_csv(index=False) 
	b64 = base64.b64encode(csv.encode('utf-8-sig')).decode()
	href = f'<a href="data:application/octet-stream;base64,{b64}" download="result_utf-8-sig.csv">Download Link</a>'
	st.markdown(f"CSVファイルのダウンロード(utf-8 BOM):  {href}", unsafe_allow_html=True)
