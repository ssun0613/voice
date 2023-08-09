# 분야 1 | 무선 기지국 장비의 통계 데이터를 활용한 인구 밀집도 예측

# 아래의 코드는 다음과 같은 순서로 구성되어 있습니다.
# 1. 데이터 로드 및 전처리
# 2. 모델 학습 및 예측
# 3. 결과 제출
# 본 코드는 결과물 제출까지의 이해를 돕기위한 코드로, 반드시 이 구조를 따를 필요는 없습니다.
# 데이터 전처리, 모델의 구성 등에서 다양한 시도를 하시되, 올바른 채점을 위한 최종 결과물의 형태에 유의하시기 바랍니다.

# 1. 데이터 로드 및 전처리
import pandas as pd

def ppr_data(df):
    # TODO: 데이터 전처리 코드 구현 ---------- #
    x_df = df.iloc[:, :-1]
    y_df = ppr_label(df)
    # ------------------------------------- #
    return x_df, y_df

def ppr_label(df):
    if 'uenomax' in df.columns:
        df = df[['datetime', 'ru_id', 'uenomax']]  # 레이블 추출
        df_pivot = df.pivot_table(index='datetime', columns='ru_id', values='uenomax', fill_value=0)  # 데이터 재구성
    else:
        df_pivot = None  # 테스트 세트의 경우 정답 컬럼 없음
    return df_pivot

train_df = pd.read_csv("Q1_train.csv")
test_df = pd.read_csv("Q1_test.csv")

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
        result_df = simplePredict(x_test)
        return result_df


def simplePredict(input_df):
    datetime_column = input_df['datetime'].drop_duplicates()  # 1. datetime 추출
    unique_ru_ids = sorted(input_df['ru_id'].unique())  # 2. unique ru_id 추출 및 정렬
    columns = ['datetime'] + list(unique_ru_ids)  # 3. 단계 1과 단계 2의 정보를 통해 컬럼 생성
    data = [[datetime] + [0 for _ in range(len(unique_ru_ids))] for datetime in
            datetime_column]  # 4. 각 타임스탬프에 대한 모든 값을 0으로 채움
    return pd.DataFrame(data, columns=columns)


model = MyModel()
model.train(x_train_df, y_train_df)
y_pred = model.predict(x_test_df)


# 3. 결과 제출
# 본 코드는 제출되는 파일의 형태에 대한 가이드로, 반드시 아래 구조를 따를 필요 없이 자유롭게 코드를 작성해도 무방합니다.
# 제출 포맷에 대해서는 data/Q1_label_sample.csv를 참조하세요.
#
# 분야 1의 제출 파일은 3개 컬럼 [datetime, BaseStationB, BaseStationJ]을 가져야 합니다.
# 각 샘플은 시간 정보, BaseStationB의 uenomax, BaseStationJ의 uenomax 값을 가져야 합니다.

def submitResult(pred):
    try:
        label = pd.read_csv('Q1_label_sample.csv')
        # 1. 컬럼명과 순서가 동일한지 체크
        if pred.columns.equals(label.columns):
            print("Check: 컬럼명과 순서가 동일합니다.")
        else:
            print(f"Warning: 컬럼명과 순서가 동일하지 않습니다.\n- 예측 데이터프레임 컬럼명: {pred.columns}\n- 레이블 데이터프레임 컬럼명: {label.columns}")
            return

        # 2. datetime 컬럼이 존재하며 해당 컬럼의 샘플수와 값이 일치하는지 체크
        if (label['datetime'] == pred['datetime']).all():
            print("Check: datetime 순서와 샘플 수가 일치합니다.")
        else:
            print("Warning: 테스트 세트와 모델 예측의 datetime이 일치하지 않습니다.")
            return

        pred.to_csv('Q1_submitResult.csv', index=False)
        print("Done : Q1_submitResult.csv 파일로 저장되었습니다.")
    except Exception as e:
        # 예외가 발생한 경우 오류 메시지 출력
        print("Error:", e)


submitResult(y_pred)