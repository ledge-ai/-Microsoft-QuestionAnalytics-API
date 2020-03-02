# -*- coding: utf-8 -*-
import os
import pandas as pd

from azure.ai.textanalytics import TextAnalyticsClient, TextAnalyticsApiKeyCredential

from wordcloud import WordCloud
from inflector import Inflector

# 入力データのパス
INPUT_DATA_PATH = 'data/sample.csv'

# 解析したい質問項目の1つ目
# サンプルでは「良かった点をご記入ください。」
FIRST_QUESTION_TEXT = '良かった点をご記入ください。'

# 解析したい質問項目の2つ目
# サンプルでは「改善が必要な点をご記入ください。」
SECOND_QUESTION_TEXT = '改善が必要な点をご記入ください。'

# Azure Text Analytics APIに必要な認証情報
TEXT_ANALYTICS_SUBSCRIPTION_KEY = '<key>'  # ご自身のSubscriptionキーを入力してください
TEXT_ANALYTICS_ENDPOINT = '<endpoint>'  # ご自身のエンドポイントURLを入力してください

# WordCloudで使用するフォントの指定
# 日本語の場合はフォントを指定する必要がある
FONT_PATH = "<pathToFont>/NotoSansCJKjp-Regular.otf"


def make_payload_list(df_question) -> (list, list):
    """
    Azure Text Analytics APIに投げるためにリストに変換。

    Args:
        df_question (pandas.DataFrame): 入力データを単一の質問に絞り込んだDataFrame
    Returns:
        payload_documents (list):       APIに投げる用のリスト
        original_position (list):       payload_documentsに対応するDataFrameの行と列を辞書型で保持するためのリスト
    """
    # 無駄に空文字をAPIに投げないようにする
    documents_list = []
    for _, row in df_question.iteritems():
        documents_list.append(row.fillna('').tolist())

    # データの整形
    payload_documents = []
    original_position = []
    for id_col, column in enumerate(documents_list):
        for id_row, text in enumerate(column):
            if text != '':
                payload_documents.append(text)
                original_position.append({
                    "column_num": id_col,
                    "index_num": id_row
                })
    return payload_documents, original_position


def fetch_sentiment_analytics_score(client, df_question, payload_list, position_list, sentiment_type) -> pd.DataFrame:
    """
    Azure Text Analytics APIにリストをPOSTしてスコアを取り出す。

    Args:
        client :                        インスタンス化されたTextAnalyticsClient
        df_question (pandas.DataFrame): 入力データを単一の質問に絞り込んだDataFrame
        payload_list (list):            APIに投げる用のリスト
        position_list (list):           payload_documentsに対応するDataFrameの行と列を辞書型で保持するためのリスト
        sentiment_type (str):           センチメント分析で取得するスコアのクラス名['positive', 'neural', 'negative']
    Return:
        df_score (pandas.DataFrame):    df_questionのテキストをスコアに置換したDataFrame
    """
    documents_length = len(payload_list)
    # 返り値をリストに結合
    merged_results = []
    STEP = 1000
    for i in range(0, documents_length, STEP):
        # 1000要素毎に Text Analytics APIに投げる
        response = client.analyze_sentiment(payload_list[i:i + STEP], language='ja')
        merged_results.extend(response)

    # スコアを取り出す
    score_list = []
    for item in merged_results:
        if not item.is_error:
            score_list.append(item['sentiment_scores'][sentiment_type])
        else:
            score_list.append('0')

    # スコアをDataFrameに格納
    df_score = df_question.copy()
    for i in range(documents_length):
        idx = position_list[i]['index_num']
        col = position_list[i]['column_num']
        # テキストをスコアに置換
        df_score.iloc[idx, col] = score_list[i]
    return df_score


def pickup_column(df_question, df_score, sentiment_type, exclude_sentences_set, thres_len=10) -> (list, str):
    """
    各カラムのセンチメント分析スコアの平均値を算出し、特定のカラムを取り出す。

    Args:
        df_question (pandas.DataFrame): 入力データを単一の質問に絞り込んだDataFrame
        df_score (pandas.DataFrame):    df_questionのテキストをスコアに置換したDataFrame
        sentiment_type (str):           センチメント分析で取得するスコアのクラス名['positive', 'neural', 'negative']
        exclude_sentences_set (set):        抽出するテキストとして取り除きたい文章を定義
        thres_len (int):                回答数が少なくなりすぎないようにしきい値を指定
    Returns:
        picked_column_list (list):      取り出したカラム内での回答テキストの配列
        col_name (str):                 取り出したカラムの名前
    """
    ave_df = df_score.mean(axis=0)
    sort_type = True if sentiment_type == 'negative' else False
    for i in range(len(ave_df)):
        # カラム名
        col_name = ave_df.sort_values(ascending=sort_type).index[i]
        # 特定カラムのリスト
        picked_column_list = df_question.loc[:, col_name].dropna().tolist()
        # 除外ワードを取り除く
        picked_column_list = [item for item in picked_column_list if item not in exclude_sentences_set]
        if len(picked_column_list) > thres_len:
            break
    # 返り値の確認
    assert len(picked_column_list) > 0 and len(col_name) > 0, \
        '返り値が正しくありません。関数への入力の確認をしてください。' \
        'picked_column_listの長さ: {0}, 取り出したカラム名: {1}'.format(len(picked_column_list), col_name)
    return picked_column_list, col_name


def normalize(txt) -> str:
    # 小文字に変換
    lowered_txt = txt.lower()
    # 単数形に変換
    normalized_txt = infr.singularize(lowered_txt)
    return normalized_txt


def preprocess(keyphrase_list, exclude_words) -> str:
    # 正規化, 除外ワードを取り除く
    normalized_list = [normalize(item) for item in keyphrase_list if item not in exclude_words]
    processed_txt = ','.join(normalized_list)
    return processed_txt


if __name__ == '__main__':
    # データの読み込み
    df_input = pd.read_csv(INPUT_DATA_PATH)

    # 結果を書き出すoutputディレクトリを作成
    os.makedirs('output', exist_ok=True)
    # 1つ目の質問を書き出す
    df_first_question = df_input.loc[:, df_input.columns.str.contains(FIRST_QUESTION_TEXT)]
    df_first_question.to_csv(os.path.join('output', 'first_question.csv'), index=None)
    print('1つ目の質問をcsvで保存しました。')

    # 2つ目の質問を書き出す
    df_second_question = df_input.loc[:, df_input.columns.str.contains(SECOND_QUESTION_TEXT)]
    df_second_question.to_csv(os.path.join('output', 'second_question.csv'), index=None)
    print('2つ目の質問をcsvで保存しました。')

    # Text Analytics APIのインスタンス化
    text_analytics_client = TextAnalyticsClient(
        endpoint=TEXT_ANALYTICS_ENDPOINT,
        credential=TextAnalyticsApiKeyCredential(TEXT_ANALYTICS_SUBSCRIPTION_KEY)
    )
    # WordCloud用の定義
    # 英語の複数形を単数形に変換するために必要
    infr = Inflector()

    # WordCloudのスタイルを定義
    wdcl = WordCloud(
        background_color='white',
        width=700, height=700,
        font_path=FONT_PATH
    )

    """ 
        ========= 1つ目の質問 ========= 
    """
    # ========= Sentiment Analytics =========
    # null以外のセルのみをリストに変換、そのときのrow, columnも辞書のリストにして保持
    first_payload_documents, first_original_position = make_payload_list(df_first_question)
    assert len(first_payload_documents) == len(first_original_position)

    # 1つ目の質問でセンチメント分析を行う
    # positiveのスコア取得
    df_first_senti_score = fetch_sentiment_analytics_score(
        client=text_analytics_client,
        df_question=df_first_question,
        payload_list=first_payload_documents,
        position_list=first_original_position,
        sentiment_type='positive'
    )

    # 除外文章を定義
    exclude_sentences = {'満足', 'やや満足', 'やや不満', '不満'}
    # 1つ目の質問に対して特定カラムとカラム名を取り出す
    first_picked_list, first_picked_col_name = pickup_column(
        df_question=df_first_question,
        df_score=df_first_senti_score,
        sentiment_type='positive',
        exclude_sentences_set=exclude_sentences
    )
    print("取り出したカラム名:", first_picked_col_name)

    # ========= KeyPhrase Extract =========
    # 抽出したカラムのリストをキーフレーズ抽出のAPIに投げる
    first_keyphrase_response = text_analytics_client.extract_key_phrases(
        inputs=first_picked_list,
        language='ja'
    )
    # レスポンスをリストに整形
    first_keyphrase_list = []
    for sentence in first_keyphrase_response:
        first_keyphrase_list.extend(sentence.key_phrases)

    # ========= Word Cloud ================
    # 除外する単語の指定 ex) {'特になし', 'なし'}
    exclude_words = {}
    # 前処理としてNanの除去と正規化（大文字→小文字、複数形→単数形）
    txt4wc = preprocess(first_keyphrase_list, exclude_words)
    wdcl.generate(text=txt4wc)

    wdcl.to_file(os.path.join('output', 'first_wordcloud.png'))
    print("1つ目の質問の作成したワードクラウドの保存完了。")

    """ 
        ========= 2つ目の質問 ========= 
    """
    # ========= Sentiment Analytics =======
    # null以外のセルのみをリストに変換、そのときのrow, columnも辞書のリストにして保持
    second_payload_documents, second_original_position = make_payload_list(df_second_question)
    assert len(second_payload_documents) == len(second_original_position)

    # 2つ目の質問でセンチメント分析を行う
    # negativeのスコア取得
    df_second_senti_score = fetch_sentiment_analytics_score(
        client=text_analytics_client,
        df_question=df_second_question,
        payload_list=second_payload_documents,
        position_list=second_original_position,
        sentiment_type='negative'
    )

    # 除外文章を定義
    exclude_sentences = {'満足', 'やや満足', 'やや不満', '不満'}
    # 2つ目の質問に対して特定カラムとカラム名を取り出す
    second_picked_list, second_picked_col_name = pickup_column(
        df_question=df_second_question,
        df_score=df_second_senti_score,
        sentiment_type='negative',
        exclude_sentences_set=exclude_sentences
    )
    print("取り出したカラム名:", second_picked_col_name)

    # ========= KeyPhrase Extract =========
    # 抽出したカラムのリストをキーフレーズ抽出のAPIに投げる
    second_keyphrase_response = text_analytics_client.extract_key_phrases(
        inputs=second_picked_list,
        language='ja'
    )
    # レスポンスをリストに整形
    second_keyphrase_list = []
    for sentence in second_keyphrase_response:
        second_keyphrase_list.extend(sentence.key_phrases)

    # ========= Word Cloud ================
    # 除外する単語の指定 ex) {'特になし', 'なし'}
    exclude_words = {}
    # 前処理としてNanの除去と正規化（大文字→小文字、複数形→単数形）
    txt4wc = preprocess(second_keyphrase_list, exclude_words)
    wdcl.generate(text=txt4wc)

    wdcl.to_file(os.path.join('output', 'second_wordcloud.png'))
    print("2つ目の質問の作成したワードクラウドの保存完了。")

