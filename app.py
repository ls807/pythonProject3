# 导入必要的库
import streamlit as st
import pandas as pd
import numpy as np
import time
import random
import shap
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
import os
from datetime import datetime
import matplotlib.font_manager as fm
from io import StringIO
import boto3

# 加载中文字体文件（需要将 SimHei.ttf 上传到项目根目录）
font_path = "SimHei.ttf"
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="贷款审批实验", page_icon=":money_with_wings:", layout="centered")

if 'page' not in st.session_state:
    st.session_state.page = 'consent'
    st.session_state.group = random.choice(['group1', 'group2', 'group3', 'group4'])
    st.session_state.case_index = 0
    st.session_state.initial_decisions = []
    st.session_state.final_decisions = []
    st.session_state.decision_times = []
    st.session_state.trust_scores = []
    st.session_state.reliance_scores = []
    st.session_state.start_time = None
    st.session_state.initial_decision_made = False
    st.session_state.final_decision_made = False
    st.session_state.sliders_completed = False
    st.session_state.results_saved = False

@st.cache_data
def load_data():
    df = pd.read_csv('loan_approval_dataset_xiu.csv')
    df.columns = df.columns.str.strip()
    df = df[['cibil_score', 'loan_term', 'loan_amount', 'income_annum', 'residential_assets_value', 'loan_status']]
    return df

df = load_data()

specific_indices = [2, 5, 1174, 10, 15, 2564, 25, 30, 1039, 40]
if 'cases' not in st.session_state:
    st.session_state.cases = df.iloc[specific_indices].reset_index(drop=True)

@st.cache_resource
def train_model(data):
    X = data.drop(['loan_status'], axis=1)
    y = data['loan_status']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    params_xgb = {
        'learning_rate': 0.02,
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'max_leaves': 127,
        'verbosity': 1,
        'seed': 42,
        'nthread': -1,
        'colsample_bytree': 0.6,
        'subsample': 0.7,
        'eval_metric': 'logloss'
    }

    model_xgb = xgb.XGBClassifier(**params_xgb)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 4],
        'learning_rate': [0.01, 0.02],
    }
    grid_search = GridSearchCV(
        estimator=model_xgb,
        param_grid=param_grid,
        scoring='neg_log_loss',
        cv=5,
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model

model = train_model(df)

@st.cache_resource
def get_shap_explainer(_model, data):
    return shap.Explainer(_model, data)

explainer = get_shap_explainer(model, df.drop('loan_status', axis=1))

def consent_page():
    st.title("知情同意书")
    st.write("""
    欢迎参加本次实验。在开始之前，请仔细阅读以下内容：

    * 本实验旨在研究人类与人工智能协作的决策过程。
    * 实验过程中，您的数据将被匿名处理，仅用于研究目的。
    * 您有权随时退出实验。

    如果您同意参加实验，请点击下方的“我同意”按钮。
    """)
    if st.button("我同意", key='consent'):
        st.session_state.page = 'instructions'

def instructions_page():
    st.title("实验说明")
    st.write("""
    **任务目标：**

    您需要扮演银行贷款审批人员，根据申请者的资料判断贷款是否会被批准。

    **实验流程：**

    1. 您将接受一个简单的培训，了解实验平台的使用以及 AI 解释。
    2. 培训结束后，需回答几个问题，全部答对后才能进入正式实验。
    3. 正式实验中，您将依次评估10个贷款申请案例。

    **报酬方式：**

    实验结束后，根据您的决策准确率发放相应的奖金，准确率越高，奖金越高，具体为准确率高于80%，会有额外的奖金。
    """)

    st.write("为了让你更好的理解实验中涉及的叠加直方图，下面将依次提供叠加直方图和数据标签的解释说明")
    st.image('叠加直方图示例.png', caption='叠加直方图示例')
    st.image('数据标签解释.png', caption='数据标签说明')

    st.write("""
    **数据分析：**

    在正式开始实验之前，请先查看以下图表，了解特征与目标变量之间的关系。
    """)

    variables = ["income_annum", "loan_term", "cibil_score", "loan_amount", "residential_assets_value"]
    target = "loan_status"

    for var in variables:
        st.write(f"**{var} 与 {target} 的关系：**")
        fig, ax = plt.subplots(figsize=(8, 5))
        for status, color in zip(df[target].unique(), ['blue', 'orange']):
            subset = df[df[target] == status]
            ax.hist(subset[var], bins=20, alpha=0.5, density=True, label=f"{target} = {status}", color=color)
        ax.set_title(f"{var} 与 {target} 的关系", fontsize=14)
        ax.set_xlabel(var, fontsize=12)
        ax.set_ylabel("密度", fontsize=12)
        ax.legend(title=target, fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        st.pyplot(fig)

    if st.button("开始培训", key='start_training'):
        st.session_state.page = 'training'

def training_page():
    st.title("培训")
    group = st.session_state.group
    st.write("""
    **报酬方式：**

    实验结束后，根据您的决策准确率发放相应的奖金，准确率越高，奖金越高，具体为准确率高于80%，会有额外的奖金。
    """)

    if group == 'group1':
        st.write("**培训内容：**")
        st.write("您将根据申请者的信息和AI的建议，判断贷款是否会被批准。")

    elif group == 'group2':
        st.write("**培训内容：**")
        st.write("您将看到AI的建议和SHAP解释图，以帮助您理解AI的决策依据。")
        st.image('shap解释.png', caption='SHAP解释图示例')

    elif group == 'group3':
        st.write("**培训内容：**")
        st.write("您将看到AI的建议和文本解释，以帮助您理解AI的决策依据。")
        st.write("""
        **文本解释示例：**

        模型认为：cibil_score 值为 700 对结果有正面影响；loan_amount 值为 500000 对结果有负面影响；loan_term 值为 15 对结果有正面影响；income_annum 值为 800000 对结果有正面影响；residential_assets_value 值为 2000000 对结果有负面影响。
        """)

    elif group == 'group4':
        st.write("**培训内容：**")
        st.write("您将看到AI的建议和交互式解释，以帮助您理解AI的决策依据。")
        st.write("以下是交互式解释的示例，您可以调整特征值，查看模型预测的变化：")
        sample_features = {}
        for col in ['cibil_score', 'loan_term', 'loan_amount', 'income_annum', 'residential_assets_value']:
            min_val = df[col].min()
            max_val = df[col].max()
            mean_val = df[col].mean()
            if col == 'loan_term':
                sample_features[col] = st.slider(f"{col}", int(min_val), int(max_val), int(mean_val), step=1, key=f'train_{col}')
            else:
                sample_features[col] = st.slider(f"{col}", float(min_val), float(max_val), float(mean_val), key=f'train_{col}')
        X_sample = pd.DataFrame(sample_features, index=[0])
        sample_prediction = model.predict(X_sample)[0]
        sample_decision = '批准' if sample_prediction == 1 else '拒绝'
        st.write(f"**模型预测结果：{sample_decision}**")

    if st.button("开始测试", key='start_quiz'):
        st.session_state.page = 'quiz'

def quiz_page():
    st.title("培训测试")
    group = st.session_state.group

    if group == 'group1':
        questions = [
            {
                'question': '在实验中，您需要根据申请者的资料和AI的建议判断贷款是否会被批准。',
                'options': ['正确', '错误'],
                'answer': '正确'
            },
        ]
    elif group == 'group2':
        questions = [
            {
                'question': '在实验中，您需要根据申请者的资料和AI的建议以及Shap解释图判断贷款是否会被批准。',
                'options': ['正确', '错误'],
                'answer': '正确'
            },
            {
                'question': '基于下方的Shap解释图，你认为redidential_assets_value是对模型预测正向影响最大的特征吗？',
                'image': '对实例的Shap解释图.png',
                'options': ['正确', '错误'],
                'answer': '正确'
            },
        ]
    elif group == 'group3':
        questions = [
            {
                'question': '在实验中，您需要根据申请者的资料和AI的建议以及文本解释判断贷款是否会被批准。',
                'options': ['正确', '错误'],
                'answer': '正确'
            },
        ]
    elif group == 'group4':
        questions = [
            {
                'question': '在实验中，您需要根据申请者的资料和AI的建议以及交互解释判断贷款是否会被批准。',
                'options': ['正确', '错误'],
                'answer': '正确'
            },
            {
                'question': '最终的预测是基于最初呈现的申请人基本信息来决定。',
                'options': ['了解', '不了解'],
                'answer': '了解'
            },
        ]

    if 'quiz_submitted' not in st.session_state:
        st.session_state.quiz_submitted = False
        st.session_state.quiz_passed = False

    if not st.session_state.quiz_submitted:
        score = 0
        total = len(questions)
        user_answers = []

        for idx, q in enumerate(questions):
            st.write(f"**问题 {idx + 1}：{q['question']}**")
            if 'image' in q:
                st.image(q['image'])
            user_answer = st.radio("", q['options'], key=f"quiz_{idx}")
            user_answers.append(user_answer)

        if st.button("提交答案", key='submit_quiz'):
            for idx, q in enumerate(questions):
                if user_answers[idx] == q['answer']:
                    score += 1
            st.session_state.quiz_submitted = True
            if score == total:
                st.session_state.quiz_passed = True
                st.success("恭喜您，全部回答正确！点击下方按钮进入正式实验。")
            else:
                st.session_state.quiz_passed = False
                st.error(f"您答对了 {score}/{total} 道题目，请重新阅读培训内容并再次尝试。")
    else:
        if st.session_state.quiz_passed:
            if st.button("进入实验", key='enter_experiment'):
                st.session_state.page = 'experiment'
        else:
            if st.button("重新培训", key='retry_training'):
                st.session_state.page = 'training'
                st.session_state.quiz_submitted = False
                st.session_state.quiz_passed = False

def experiment_page():
    st.title(f"实验进行中：案例 {st.session_state.case_index + 1}/10")
    case = st.session_state.cases.iloc[st.session_state.case_index]
    st.write("**申请者信息：**")
    st.table(case.drop('loan_status').to_frame().T)

    if not st.session_state.initial_decision_made:
        st.write("请根据以上信息判断贷款是否会被批准：")
        initial_decision = st.radio("", ['批准', '拒绝'], key=f'initial_decision_{st.session_state.case_index}')
        if st.button("提交初始决策", key=f'submit_initial_{st.session_state.case_index}'):
            st.session_state.initial_decisions.append(initial_decision)
            st.session_state.initial_decision_made = True
            st.session_state.start_time = time.time()
    elif not st.session_state.final_decision_made:
        st.write(f"**您的初始决策：{st.session_state.initial_decisions[-1]}**")
        X_case = case.drop('loan_status').to_frame().T
        ai_prediction = model.predict(X_case)[0]
        ai_decision = '批准' if ai_prediction == 1 else '拒绝'
        st.write(f"**AI 建议：{ai_decision}**")

        group = st.session_state.group

        if group == 'group1':
            st.write("（此组别不提供AI解释。）")
        elif group == 'group2':
            shap_values = explainer(X_case)
            st.write("**AI 解释（SHAP 解释）：**")
            shap.initjs()
            fig, ax = plt.subplots()
            shap.plots.waterfall(shap_values[0], max_display=5, show=False)
            st.pyplot(fig)
        elif group == 'group3':
            st.write("**AI 解释（文本解释）：**")
            shap_values = explainer(X_case)
            feature_impact = pd.DataFrame({
                'feature': X_case.columns,
                'value': X_case.values[0],
                'shap_value': shap_values.values[0]
            })
            feature_impact['abs_shap'] = feature_impact['shap_value'].abs()
            feature_impact.sort_values('abs_shap', ascending=False, inplace=True)
            explanation = "模型认为："
            first = True
            for idx, row in feature_impact.iterrows():
                if first:
                    first = False
                else:
                    explanation += "；"
                if row['shap_value'] > 0:
                    explanation += f"{row['feature']} 值为 {row['value']} 对结果有正面影响"
                else:
                    explanation += f"{row['feature']} 值为 {row['value']} 对结果有负面影响"
            st.write(explanation)
        elif group == 'group4':
            st.write("**AI 解释（交互式解释）：**")
            st.write("您可以调整以下特征值，查看模型预测的变化：")
            adjusted_features = {}
            for col in X_case.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                mean_val = float(X_case[col])
                if col == 'loan_term':
                    adjusted_value = st.slider(f"{col}", int(min_val), int(max_val), int(mean_val), key=f'adjust_{col}_{st.session_state.case_index}', step=1)
                else:
                    adjusted_value = st.slider(f"{col}", float(min_val), float(max_val), float(mean_val), key=f'adjust_{col}_{st.session_state.case_index}')
                adjusted_features[col] = adjusted_value
            X_adjusted = pd.DataFrame(adjusted_features, index=[0])
            adjusted_prediction = model.predict(X_adjusted)[0]
            adjusted_decision = '批准' if adjusted_prediction == 1 else '拒绝'
            st.write(f"**调整后的AI建议：{adjusted_decision}**")

        final_decision = st.radio("请给出您的最终决策：", ['批准', '拒绝'], key=f'final_decision_{st.session_state.case_index}')
        if st.button("提交最终决策", key=f'submit_final_{st.session_state.case_index}'):
            st.session_state.final_decisions.append(final_decision)
            st.session_state.final_decision_made = True
            decision_time = time.time() - st.session_state.start_time
            st.session_state.decision_times.append(decision_time)
    elif not st.session_state.sliders_completed:
        st.write("**请回答以下问题（0-100）：**")
        trust_score = st.slider("我完全相信AI预测：", 0, 100, key=f'trust_score_{st.session_state.case_index}')
        reliance_score = st.slider("我依赖于AI的提示：", 0, 100, key=f'reliance_score_{st.session_state.case_index}')
        if st.button("下一步", key=f'next_{st.session_state.case_index}'):
            st.session_state.trust_scores.append(trust_score)
            st.session_state.reliance_scores.append(reliance_score)
            st.session_state.sliders_completed = True

            st.session_state.case_index += 1
            st.session_state.initial_decision_made = False
            st.session_state.final_decision_made = False
            st.session_state.sliders_completed = False
            st.session_state.start_time = None

            if st.session_state.case_index >= len(st.session_state.cases):
                st.session_state.page = 'survey'
    else:
        st.write("请按照指示完成实验步骤。")

def survey_page():
    st.title("问卷调查")
    st.write("感谢您完成实验。请回答以下问题，其中“非常同意”、“同意”、“有点同意”、“中立”、“有点不同意”、“不同意”和“非常不同意”，分别记为7、6、5、4、3、2、1。")

    questions = [
        "1. 系统提供了有用的信息来了解申请者的贷款审批情况。",
        "2. 系统为申请者的贷款审批情况提供了新的见解。",
        "3. 系统帮助我思考并以更少的努力完成审批任务。",
        "4. 我依赖系统的分析进行最终审批。",
        "5. 我可以信任系统提供的结果或/和解释。",
        "6. 我在使用系统时感到不安全、沮丧和压力。",
        "7. 我将使用该系统来审批申请人的贷款情况。",
        "8. 我认为AI与数据驱动工具可以改善贷款审批。"
    ]

    responses = []

    options = {
        7: '非常同意',
        6: '同意',
        5: '有点同意',
        4: '中立',
        3: '有点不同意',
        2: '不同意',
        1: '非常不同意'
    }

    for idx, question in enumerate(questions):
        st.write(question)
        response = st.radio("", list(options.values()), key=f'survey_q{idx+1}')
        score = [k for k, v in options.items() if v == response][0]
        responses.append(score)

    if st.button("提交问卷", key='submit_survey'):
        st.session_state.survey_responses = responses
        st.session_state.page = 'thankyou'

def thankyou_page():
    st.title("实验结束")
    st.write("感谢您的参与！")

    num_cases = len(st.session_state.initial_decisions)
    results = pd.DataFrame({
        'Case_Index': list(range(1, num_cases + 1)),
        'Initial_Decision': st.session_state.initial_decisions,
        'Final_Decision': st.session_state.final_decisions,
        'Decision_Time': st.session_state.decision_times,
        'Trust_Score': st.session_state.trust_scores,
        'Reliance_Score': st.session_state.reliance_scores
    })

    survey_df = pd.DataFrame({
        'Question': [f"Q{idx+1}" for idx in range(len(st.session_state.survey_responses))],
        'Response': st.session_state.survey_responses
    })

    st.write("您的实验结果：")
    st.dataframe(results)
    st.write("您的问卷调查回答：")
    st.dataframe(survey_df)

    # 使用 st.secrets 中的 AWS 凭证信息
    aws_access_key = st.secrets["aws"]["aws_access_key_id"]
    aws_secret_key = st.secrets["aws"]["aws_secret_access_key"]
    bucket_name = st.secrets["aws"]["aws_bucket_name"]
    region_name = st.secrets["aws"]["aws_region"]

    s3 = boto3.client('s3',
                      aws_access_key_id=aws_access_key,
                      aws_secret_access_key=aws_secret_key,
                      region_name=region_name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    participant_id = f"participant_{timestamp}"
    folder_path = f"{st.session_state.group}/{participant_id}"

    results_csv = results.to_csv(index=False)
    survey_csv = survey_df.to_csv(index=False)

    s3.put_object(Bucket=bucket_name, Key=f"{folder_path}/results.csv", Body=results_csv)
    s3.put_object(Bucket=bucket_name, Key=f"{folder_path}/survey.csv", Body=survey_csv)

    st.write("实验数据已上传至 S3 存储桶。")
    st.write("请注意，您可以使用 S3 控制台或相应的工具从 S3 下载这些文件。")

# 页面路由
if st.session_state.page == 'consent':
    consent_page()
elif st.session_state.page == 'instructions':
    instructions_page()
elif st.session_state.page == 'training':
    training_page()
elif st.session_state.page == 'quiz':
    quiz_page()
elif st.session_state.page == 'experiment':
    experiment_page()
elif st.session_state.page == 'survey':
    survey_page()
elif st.session_state.page == 'thankyou':
    thankyou_page()
else:
    st.session_state.page = 'consent'
    consent_page()












