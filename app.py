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
import boto3

# 使用微软雅黑字体确保中英文与负号正确显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑
plt.rcParams['axes.unicode_minus'] = True  # 确保负号显示

# 配置Streamlit页面
st.set_page_config(page_title="贷款审批实验", page_icon=":money_with_wings:", layout="centered")

# 初始化会话状态
if 'page' not in st.session_state:
    st.session_state.page = 'consent'
    st.session_state.group = random.choice(['group1', 'group2', 'group3', 'group4'])
    st.session_state.case_index = 0
    st.session_state.final_decisions = []
    st.session_state.decision_times = []
    st.session_state.trust_scores = []
    st.session_state.reliance_scores = []
    st.session_state.start_time = None
    st.session_state.results_saved = False
    st.session_state.pre_survey_responses = {}
    st.session_state.survey_responses = []
    st.session_state.quiz_submitted = False
    st.session_state.quiz_passed = False
    st.session_state.current_step = 1
    st.session_state.instructions_start_time = None
    st.session_state.show_warning = False
    st.session_state.show_read_options = False  # 新增用于控制是否显示阅读确认按钮

@st.cache_data
def load_data():
    df = pd.read_csv('loan_approval_dataset_xiu.csv')
    df.columns = df.columns.str.strip()
    # 重命名列名为中文并筛选指定列
    df.rename(columns={
        'income_annum': '年收入',
        'loan_amount': '贷款金额',
        'loan_term': '贷款期限',
        'cibil_score': '信用评分',
        'residential_assets_value': '住房资产价值',
        'loan_status': '贷款申请状态'
    }, inplace=True)
    df = df[['年收入', '贷款金额', '贷款期限', '信用评分', '住房资产价值', '贷款申请状态']]
    return df

df = load_data()

indices_first_10 = [70, 90, 2856, 100, 564, 110, 120, 3474, 130, 140]  # 前10无解释
indices_last_10 = [5, 10, 1039, 15, 1174, 25, 30, 2564, 40, 50]       # 后10有解释
specific_indices = indices_first_10 + indices_last_10

if 'cases' not in st.session_state:
    st.session_state.cases = df.iloc[specific_indices].reset_index(drop=True)

@st.cache_resource
def train_model(data):
    X = data.drop(['贷款申请状态'], axis=1)
    y = data['贷款申请状态']
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

explainer = get_shap_explainer(model, df.drop('贷款申请状态', axis=1))

def consent_page():
    st.title("知情同意书")
    st.write("""
    欢迎参加本次实验。在开始之前，请仔细阅读以下内容：

    * 本实验旨在研究人类与人工智能协作的决策过程。
    * 实验过程中，您的数据将被匿名处理，仅用于研究目的。
    * 请按照要求进行真实的回答，实验过程可能会存在轻微卡顿情况，还请谅解。
    * 您有权随时退出实验。

    如果您同意参加实验，请点击下方的“我同意”按钮。
    """)
    if st.button("我同意", key='consent'):
        st.session_state.page = 'instructions'

def instructions_page():
    if st.session_state.instructions_start_time is None:
        st.session_state.instructions_start_time = time.time()

    st.title("实验说明")
    st.write("""
    **任务目标：**

    您需要扮演银行贷款审批人员，根据申请者的资料判断贷款是否会被批准。

    **实验流程：**

    1. 您将接受一个简单的培训，了解实验平台的使用以及 AI 解释。
    2. 培训结束后，需回答几个问题，全部答对后才能进入正式实验。
    3. 正式实验中，您将依次评估20个贷款申请案例（前10个无解释，后10个视组别而定）。
    4. 实验中模型预测的准确率控制在70%。

    **报酬方式：**

    实验结束后，将根据您的决策准确性分发奖金，除基础奖金外，准确率高于85%的参与者将获得额外奖金。

    **相关解释说明：**

    本实验中的AI解释包括SHAP解释、文本解释以及交互式解释（视组别而定），以帮助您了解模型决策的依据。

    **解释说明：**

    在正式开始实验之前，请先查看以下图表，了解特征与目标变量之间的关系。
    """)

    st.write("为了让你更好的理解实验中涉及的叠加直方图，下面将依次提供叠加直方图和数据标签的解释说明")
    st.image('叠加直方图示例.png', caption='叠加直方图示例')
    st.image('数据标签解释.png', caption='数据标签说明')
    st.write("提醒：后续呈现的申请者信息标签均为上述英文形式，请记住其代表的中文含义。")

    variables = ["年收入", "贷款期限", "信用评分", "贷款金额", "住房资产价值"]
    target = "贷款申请状态"

    for var in variables:
        st.write(f"**{var} 与 {target} 的关系：**")
        fig, ax = plt.subplots(figsize=(8, 5))
        for status, color in zip(df[target].unique(), ['blue', 'orange']):
            subset = df[df[target] == status]
            ax.hist(subset[var], bins=20, alpha=0.5, density=True, label=f"{target} = {status}", color=color)
        ax.set_title(f"{var} 与 {target} 的关系", fontsize=14)
        ax.set_xlabel(var, fontsize=12)
        ax.set_ylabel("密度", fontsize=12)
        ax.legend(title=target, fontsize=10, loc='upper right')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        st.pyplot(fig)

    browse_time = time.time() - st.session_state.instructions_start_time

    if st.button("如果您已了解实验情况，请点击开始测试", key='start_test'):
        if browse_time < 60:
            st.session_state.show_read_options = True
        else:
            st.session_state.page = 'training'

    if st.session_state.show_read_options:
        st.warning("您是否已认真阅读了实验说明部分？")
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("我已充分阅读", key='fully_read'):
                st.session_state.page = 'training'
                st.session_state.show_read_options = False
        with col_no:
            if st.button("我还需继续阅读", key='need_more_time'):
                st.session_state.instructions_start_time = time.time()
                st.session_state.show_read_options = False
        # 不需要调用 st.stop()，让 Streamlit 自然重新运行脚本

def training_page():
    st.title("培训")
    group = st.session_state.group

    st.write("""
    **报酬方式：**

    实验结束后，将根据您的决策准确性分发奖金，除基础奖金外，准确率高于85%的参与者将获得额外奖金。
    """)

    if group == 'group1':
        st.write("您将根据申请者的信息和AI的建议，判断贷款是否会被批准。（该组不提供AI解释）")
    elif group == 'group2':
        st.write("您将根据申请者信息和AI的建议以及SHAP解释图，判断贷款是否会被批准")
        st.image('Shap解释说明.png', caption='SHAP解释图示例')
        st.write("由以上的shap瀑布图可知，模型根据涉及的特征计算得到的f(x)值为2.011，大于0，预测结果为不患心脏病（示例）")
    elif group == 'group3':
        st.write("您将根据申请者信息和AI的建议以及文本解释，判断贷款是否会被批准")
        st.write("""
        **文本解释示例：**

        模型认为：特征1 值为 700 对结果有正面影响；特征2 值为 500000 对结果有负面影响；特征3 值为 15 对结果有正面影响；特征4 值为 800000 对结果有正面影响；特征5 值为 2000000 对结果有负面影响。
        """)
        st.write("特征的顺序是按对模型预测重要性大小排序，重要性高的特征排在前面。")
    elif group == 'group4':
        st.write("您将根据申请者信息和AI的建议以及交互解释，判断贷款是否会被批准")
        st.write("以下是交互说明示例（培训用）：")
        st.image('交互说明.png', caption='交互说明')

    if st.button("如果您已了解实验情况，请点击开始测试", key='start_quiz'):
        st.session_state.page = 'quiz'

def quiz_page():
    st.title("培训测试")
    group = st.session_state.group

    # 定义注意力测试题
    attention_question = {
        'question': '可用于照明的电器是？',
        'options': ['音响', '台灯', '空调', '洗衣机'],
        'answer': '台灯'
    }

    if group == 'group1':
        questions = [
            {
                'question': '在实验中，您需要根据申请者的资料和AI的建议判断贷款是否会被批准。',
                'options': ['正确', '错误'],
                'answer': '正确'
            },
            attention_question  # 添加注意力测试题
        ]
    elif group == 'group2':
        questions = [
            {
                'question': '在实验中，您需要根据申请者的资料和AI的建议以及Shap解释图判断贷款是否会被批准。',
                'options': ['正确', '错误'],
                'answer': '正确'
            },
            {
                'question': '基于下方的Shap解释图，你认为下面哪个是对模型预测正向影响最大的特征？',
                'image': '培训测试.png',
                'options': ['ca', 'chol', 'fbs', 'thal'],
                'answer': 'thal'
            },
            attention_question  # 添加注意力测试题
        ]
    elif group == 'group3':
        questions = [
            {
                'question': '在实验中，您需要根据申请者的资料和AI的建议以及文本解释判断贷款是否会被批准。',
                'options': ['正确', '错误'],
                'answer': '正确'
            },
            {
                'question': '在本实验中，文本解释是用来帮助您理解人工智能预测结果的主要依据。以下哪项描述正确反映了文本解释的特征排序规则？',
                'options': [
                    '特征按照字母顺序排序',
                    '特征按照它们对预测结果的重要性从高到低排序',
                    '特征按照数据输入的时间顺序排序',
                    '特征按照系统随机顺序显示'
                ],
                'answer': '特征按照它们对预测结果的重要性从高到低排序'
            },
            attention_question  # 添加注意力测试题
        ]
    elif group == 'group4':
        questions = [
            {
                'question': '在实验中，您需要根据申请者的资料和AI的建议以及交互解释判断贷款是否会被批准。',
                'options': ['正确', '错误'],
                'answer': '正确'
            },
            {
                'question': '交互式解释的主要目的是：',
                'options': [
                    '提供模型预测的总体准确率',
                    '允许用户通过调整特征值观察预测结果的变化',
                    '按特征重要性从高到低展示SHAP值',
                    '显示AI的内部计算过程'
                ],
                'answer': '允许用户通过调整特征值观察预测结果的变化'
            },
            {
                'question': '在交互式解释中，如果某个特征值的调整对预测结果没有任何变化，这可能意味着：',
                'options': [
                    '预测模型无法识别该特征',
                    '特征值的调整范围设置有误',
                    '预测模型存在重大错误',
                    '该特征对模型预测结果的影响较小'
                ],
                'answer': '该特征对模型预测结果的影响较小'
            },
            attention_question  # 添加注意力测试题
        ]

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
                st.success("恭喜您，全部回答正确！点击下方按钮进入正式实验前的调查。")
            else:
                st.session_state.quiz_passed = False
                st.error(f"您答对了 {score}/{total} 道题目，请重新阅读培训内容并再次尝试。")
    else:
        if st.session_state.quiz_passed:
            if st.button("进行实验前调查", key='enter_pre_survey'):
                st.session_state.page = 'pre_experiment_survey'
        else:
            if st.button("你未通过测试，请重新阅读培训内容", key='retry_training'):
                st.session_state.page = 'training'
                st.session_state.quiz_submitted = False
                st.session_state.quiz_passed = False

def pre_experiment_survey_page():
    st.title("正式实验前的调查")

    # 1.请选择您的性别
    gender = st.radio("1. 请选择您的性别？", ["男", "女"], key='gender')  # 移除“其他”

    # 2.您的年龄为？
    age = st.text_input("2. 您的年龄为？", key='age')

    # 3. 您对人工智能的了解程度为？
    ai_knowledge = st.radio(
        "3. 您对人工智能的了解程度为？",
        [
            "我不了解人工智能",
            "我知道人工智能的基本概念",
            "我使用过人工智能",
            "我是人工智能方面的专家"
        ],
        key='ai_knowledge'
    )

    st.write("请回答下列量表的题项，所有题项采用李克特5分量表（1、2、3、4、5分别表示“非常不同意”、“不同意”、“不确定”、“同意”、“非常同意”）")

    cognitive_style_questions = [
        "在做决定之前，我通常会仔细思考。",
        "我会认真倾听自己的内心感受。",
        "在做决定之前，我会先思考好想要实现的目标。",
        "在大多数情况下，依赖自己的感觉做决定是明智的。",
        "我不喜欢需要依靠直觉做决定的情境。",
        "我常常反思自己。",
        "我喜欢制定周密的计划，而不是听天由命。",
        "我更喜欢根据自己的感觉、人际理解和生活经验来做出判断。",
        "我的感觉在决策中起着重要作用。",
        "我是个完美主义者。",
        "当我需要证明某个决定的合理性时，我会特别仔细地思考。",
        "在决定是否信任他人方面，我通常会依赖自己的直觉。",
        "面对问题时，我会先分析清楚事实和细节，然后再做决定。",
        "我会先思考，然后再行动。",
        "我更喜欢感性的人。",
        "与他人相比，我更认真地考虑自己的计划和目标。",
        "当所有备选方案都差不多时，我会选择让我感觉最舒服的那一个。",
        "我是一个非常凭直觉的人。",
        "我喜欢情感丰富的场景、讨论和电影。"
    ]

    pre_survey_responses = {
        'gender': gender,
        'age': age,
        'ai_knowledge': ai_knowledge
    }

    for i, question in enumerate(cognitive_style_questions):
        score = st.slider(f"第{i+1}题：{question}", 1, 5, 1, key=f'pre_style_{i+1}')
        pre_survey_responses[f"cognitive_style_q{i+1}"] = score

    # 计算直觉型得分和和深思熟虑型得分和
    intuitive_indices = [2, 4, 5, 8, 9, 12, 15, 17, 18, 19]       # 直觉型题目编号
    deliberative_indices = [1, 3, 6, 7, 10, 11, 13, 14, 16]    # 深思熟虑型题目编号

    intuitive_sum = sum([pre_survey_responses[f"cognitive_style_q{idx}"] for idx in intuitive_indices])
    deliberative_sum = sum([pre_survey_responses[f"cognitive_style_q{idx}"] for idx in deliberative_indices])

    pre_survey_responses['直觉型得分和'] = intuitive_sum
    pre_survey_responses['深思熟虑型得分和'] = deliberative_sum

    if st.button("提交调查", key='submit_pre_survey'):
        # 直接存储调查数据，包括计算的得分和
        st.session_state.pre_survey_responses = pre_survey_responses
        st.session_state.page = 'experiment'

def experiment_page():
    st.title(f"实验进行中：案例 {st.session_state.case_index + 1}/{len(st.session_state.cases)}")

    if st.session_state.start_time is None:
        st.session_state.start_time = time.time()

    case = st.session_state.cases.iloc[st.session_state.case_index]
    st.write("**申请者信息：**")
    # 使用 dataframe 并 use_container_width=True 使内容宽度扩大
    st.dataframe(case.drop('贷款申请状态').to_frame().T, use_container_width=True)

    no_explain = True if st.session_state.case_index < 10 else False
    X_case = case.drop('贷款申请状态').to_frame().T
    ai_prediction = model.predict(X_case)[0]
    ai_decision = '批准' if ai_prediction == 1 else '拒绝'
    st.write(f"**AI 建议：{ai_decision}**")

    group = st.session_state.group
    if no_explain:
        st.write("(此案例不提供任何AI解释。)")
    else:
        if group == 'group1':
            st.write("（该组不提供AI解释。）")
        elif group == 'group2':
            shap_values = explainer(X_case)
            st.write("**AI 解释（SHAP 解释）：**")
            shap.initjs()
            fig, ax = plt.subplots()
            shap.plots.waterfall(shap_values[0], max_display=5, show=False)
            st.pyplot(fig)
        elif group == 'group3':
            shap_values = explainer(X_case)
            st.write("**AI 解释（文本解释）：**")
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
                adjusted_value = st.slider(col, float(min_val), float(max_val), float(mean_val), key=f'adjust_{col}_{st.session_state.case_index}')
                if col == '贷款期限':
                    adjusted_value = int(adjusted_value)
                adjusted_features[col] = adjusted_value
            X_adjusted = pd.DataFrame(adjusted_features, index=[0])
            adjusted_prediction = model.predict(X_adjusted)[0]
            adjusted_decision = '批准' if adjusted_prediction == 1 else '拒绝'
            st.write(f"**调整后的AI建议：{adjusted_decision}**")

    if st.session_state.current_step == 1:
        with st.form(key='decision_form'):
            final_decision = st.radio("请给出您的最终决策：", ['批准', '拒绝'], key=f'final_decision_{st.session_state.case_index}')
            submit_decision = st.form_submit_button(label='提交决策')

        if submit_decision:
            st.session_state.final_decisions.append(final_decision)
            decision_time = time.time() - st.session_state.start_time
            st.session_state.decision_times.append(decision_time)
            st.session_state.current_step = 2

    elif st.session_state.current_step == 2:
        with st.form(key='scores_form'):
            st.write("**请回答以下问题（0-100）：**")

            st.write("你相信人工智能提供的决策及依据吗？")
            trust_score = st.slider("", 0, 100, key=f'trust_score_{st.session_state.case_index}')
            col1, col2 = st.columns([1,1])
            with col1:
                st.write("0 ----- 完全不相信")
            with col2:
                st.write("100 ----- 完全相信")

            st.write("你在决策时所采取的策略为？")
            reliance_score = st.slider("", 0, 100, key=f'reliance_score_{st.session_state.case_index}')
            col3, col4 = st.columns([1,1])
            with col3:
                st.write("0 ----- 完全依赖于自己数据分析")
            with col4:
                st.write("100 ----- 完全依赖于AI的提示")

            submit_scores = st.form_submit_button(label='提交评分')

        if submit_scores:
            st.session_state.trust_scores.append(trust_score)
            st.session_state.reliance_scores.append(reliance_score)
            st.session_state.case_index += 1
            st.session_state.current_step = 1
            st.session_state.start_time = None

            if st.session_state.case_index >= len(st.session_state.cases):
                st.session_state.page = 'survey'
            else:
                st.session_state.page = 'experiment'

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

    with st.form(key='survey_form'):
        for idx, question in enumerate(questions):
            st.write(question)
            response = st.radio("", list(options.values()), key=f'survey_q{idx+1}')
            score = [k for k, v in options.items() if v == response][0]
            responses.append(score)

        submit_survey = st.form_submit_button(label='提交问卷')

    if submit_survey:
        st.session_state.survey_responses = responses
        st.session_state.page = 'thankyou'

def thankyou_page():
    st.title("实验结束")
    st.write("感谢您的参与！")

    num_cases = len(st.session_state.final_decisions)
    results = pd.DataFrame({
        'Case_Index': list(range(1, num_cases + 1)),
        'Final_Decision': st.session_state.final_decisions,
        'Decision_Time': st.session_state.decision_times,
        'Trust_Score': st.session_state.trust_scores,
        'Reliance_Score': st.session_state.reliance_scores
    })

    survey_df = pd.DataFrame({
        'Question': [f"Q{idx+1}" for idx in range(len(st.session_state.survey_responses))],
        'Response': st.session_state.survey_responses
    })

    pre_survey_df = pd.DataFrame([st.session_state.pre_survey_responses])

    st.write("您的实验结果：")
    st.dataframe(results)
    st.write("您的问卷调查回答：")
    st.dataframe(survey_df)
    st.write("您的实验前调查回答：")
    st.dataframe(pre_survey_df)

    try:
        aws_access_key = st.secrets["aws"]["aws_access_key_id"]
        aws_secret_key = st.secrets["aws"]["aws_secret_access_key"]
        bucket_name = st.secrets["aws"]["aws_bucket_name"]
        region_name = st.secrets["aws"]["aws_region"]

        s3 = boto3.client('s3',
                          aws_access_key_id=aws_access_key,
                          aws_secret_access_key=aws_secret_key,
                          region_name=region_name)

        group = st.session_state.group
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        participant_id = f"participant_{timestamp}"
        folder_path_s3 = f"{group}/{participant_id}"
        results_csv = results.to_csv(index=False)
        survey_csv = survey_df.to_csv(index=False)
        pre_survey_csv = pre_survey_df.to_csv(index=False)

        s3.put_object(Bucket=bucket_name, Key=f"{folder_path_s3}/results.csv", Body=results_csv)
        s3.put_object(Bucket=bucket_name, Key=f"{folder_path_s3}/survey.csv", Body=survey_csv)
        s3.put_object(Bucket=bucket_name, Key=f"{folder_path_s3}/pre_survey.csv", Body=pre_survey_csv)

        st.success("实验数据已上传至 S3 存储桶。")
        st.write("请注意，您可以使用 S3 控制台或相应的工具从 S3 下载这些文件。")
    except Exception as e:
        st.error(f"上传到S3时出错: {e}")

def main():
    if st.session_state.page == 'consent':
        consent_page()
    elif st.session_state.page == 'instructions':
        instructions_page()
    elif st.session_state.page == 'training':
        training_page()
    elif st.session_state.page == 'quiz':
        quiz_page()
    elif st.session_state.page == 'pre_experiment_survey':
        pre_experiment_survey_page()
    elif st.session_state.page == 'experiment':
        experiment_page()
    elif st.session_state.page == 'survey':
        survey_page()
    elif st.session_state.page == 'thankyou':
        thankyou_page()
    else:
        st.session_state.page = 'consent'
        consent_page()

if __name__ == "__main__":
    main()











