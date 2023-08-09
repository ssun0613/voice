# 분야 2 | 유선 네트워크 장비의 신속한 조치를 위한 경보 유형 분류

# 아래의 코드는 다음과 같은 순서로 구성되어 있습니다.
# 1. 데이터 로드 및 전처리
# 2. 모델 학습 및 예측
# 3. 결과 제출
# 본 코드는 결과물 제출까지의 이해를 돕기위한 코드로, 반드시 이 구조를 따를 필요는 없습니다.
# 데이터 전처리, 모델의 구성 등에서 다양한 시도를 하시되, 올바른 채점을 위한 최종 결과물의 형태에 유의하시기 바랍니다.

# 1. 데이터 로드
import pandas as pd


def ppr_data(df):
    # TODO: 데이터 전처리 코드 구현 ---------- #
    df = df.groupby('ticketno').apply(custom_info).reset_index()  # ticketno 기준 병합 및 전처리 수행

    x_df = df.iloc[:, :-1]
    y_df = df['root_cause_type']

    # ------------------------------------- #
    return x_df, y_df


def custom_info(group):  # 임의로 작성된 전처리 코드입니다.
    d = {}
    group.sort_values(by='alarmtime', ascending=True, inplace=True)  # 경보 순서 정렬
    d['alarmmsg_original'] = ' '.join(group['alarmmsg_original'])  # 메시지 단순 병합
    if 'root_cause_type' in group.columns:  # 레이블 추출
        d['root_cause_type'] = group['root_cause_type'].iloc[0]  # 동일한 ticketno는 동일한 root_cause_type을 가짐
    else:
        d['root_cause_type'] = None  # 테스트 세트의 경우 정답 컬럼 없음
    return pd.Series(d, index=['alarmmsg_original', 'root_cause_type'])


train_df = pd.read_csv("Q2_train.csv")
test_df = pd.read_csv("Q2_test.csv")

x_train_df, y_train_df = ppr_data(train_df)
x_test_df, _ = ppr_data(test_df)  # 테스트 세트의 경우 정답 컬럼 없음


# 2. 모델 학습 및 예측
class MyModel:
    def __init__(self) -> None:
        self.model = None

    def train(self, x_train):
        # TODO: 모델 학습 코드 구현 ---------- #

        # --------------------------------- #
        pass

    def predict(self, x_test):
        # 1. ticketno 컬럼은 입력받은 값으로 채우고,
        # 2. pred 컬럼은 모두 'LinkCut' 값으로 채운 데이터프레임 생성
        pred_df = pd.DataFrame({'ticketno': x_test['ticketno'].values, 'root_cause_type': ['LinkCut'] * len(x_test)})
        return pred_df


model = MyModel()
model.train(x_train_df, y_train_df)
y_pred = model.predict(x_test_df)


# 3. 결과 제출
# 본 코드는 제출되는 파일의 형태에 대한 가이드로, 반드시 아래 구조를 따를 필요 없이 자유롭게 코드를 작성해도 무방합니다.
# 제출 포맷에 대해선 data/Q2_label_sample.csv를 참조하세요.
#
# 분야 2의 경우, 전표(ticket) 하나에 하나의 근원장애(root_cause_type)을 매칭해야 합니다.
#   주의: 경보(alarm) 개수와 전표(ticket) 개수는 다르며, 예측할 대상은 전표입니다.
#   주의: ticketno 컬럼 기준으로 오름차순 정렬이 필요합니다.
# 분야 2의 제출 파일은 2개 컬럼 [ticketno, root_cause_type]을 가져야 합니다.

def submitResult(pred):
    try:
        label = pd.read_csv('Q2_label_sample.csv')
        # ticketno 순서와 개수가 일치하는지 확인
        if (label['ticketno'] == pred['ticketno']).all():
            print("Check: ticketno 순서와 샘플 수가 일치합니다.")
        else:
            print("Warning: 테스트 세트와 모델 예측의 ticketno가 일치하지 않습니다.")
            return

        pred.to_csv('Q2_submitResult.csv', index=False)
        print("Done : Q2_submitResult.csv 파일로 저장되었습니다.")
    except Exception as e:
        # 예외가 발생한 경우 오류 메시지 출력
        print("Error:", e)


submitResult(y_pred)