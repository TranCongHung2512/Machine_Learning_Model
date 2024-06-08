import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, \
    GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle


def main():
    st.title(":robot_face: Machine Learning Model")

    st.header("1. Upload Data")

    uploaded_file = st.file_uploader("Chọn file data")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Dataset:")
        st.write(data.head())

        st.write("Mô tả data:")
        st.write(data.describe())

        st.header("2. Trực quan dữ liệu")

        # Chọn cột để vẽ biểu đồ
        selected_column = st.selectbox("Chọn cột muốn vẽ", data.columns)

        # Chọn kiểu biểu đồ
        plot_type = st.selectbox(
            "Chọn loại",
            ("Histogram", "Box Plot", "Scatter Plot", "Line Plot", "Distribution Plot", "Displot", "Heatmap",
             "Barh Plot")
        )

        if plot_type == "Histogram":
            plt.figure(figsize=(10, 6))
            plt.hist(data[selected_column], bins=30, edgecolor='k')
            plt.xlabel(selected_column)
            plt.ylabel('Frequency')
            plt.title(f'Histogram of {selected_column}')
            st.pyplot(plt)

        elif plot_type == "Box Plot":
            plt.figure(figsize=(10, 6))
            sns.boxplot(data[selected_column])
            plt.xlabel(selected_column)
            plt.title(f'Box Plot of {selected_column}')
            st.pyplot(plt)

        elif plot_type == "Scatter Plot":
            second_column = st.selectbox("Select second column for scatter plot", data.columns)
            plt.figure(figsize=(10, 6))
            plt.scatter(data[selected_column], data[second_column])
            plt.xlabel(selected_column)
            plt.ylabel(second_column)
            plt.title(f'Scatter Plot of {selected_column} vs {second_column}')
            st.pyplot(plt)

        elif plot_type == "Line Plot":
            second_column = st.selectbox("Select second column for line plot", data.columns)
            plt.figure(figsize=(10, 6))
            plt.plot(data[selected_column], data[second_column])
            plt.xlabel(selected_column)
            plt.ylabel(second_column)
            plt.title(f'Line Plot of {selected_column} vs {second_column}')
            st.pyplot(plt)

        elif plot_type == "Distribution Plot":
            plt.figure(figsize=(10, 6))
            sns.histplot(data[selected_column], kde=True)
            plt.xlabel(selected_column)
            plt.title(f'Distribution Plot of {selected_column}')
            st.pyplot(plt)

        elif plot_type == "Displot":
            plt.figure(figsize=(10, 6))
            sns.displot(data[selected_column], kde=True)
            plt.xlabel(selected_column)
            plt.title(f'Displot of {selected_column}')
            st.pyplot(plt)

        elif plot_type == "Heatmap":
            categorical = data.select_dtypes(include=['object']).columns

            # Áp dụng One-Hot Encoding
            data_plot = pd.get_dummies(data, columns=categorical, drop_first=True)

            plt.figure(figsize=(10, 6))
            sns.heatmap(data_plot.corr(), annot=True, cmap='coolwarm')
            plt.title('Correlation Heatmap')
            st.pyplot(plt)

        elif plot_type == "Barh Plot":
            x_column = st.selectbox("Select x-axis column", data.columns)
            y_column = st.selectbox("Select y-axis column", data.columns)
            plt.figure(figsize=(10, 6))
            sns.barplot(x=data[x_column], y=data[y_column])
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            plt.title(f'Barh Plot of {x_column} vs {y_column}')
            st.pyplot(plt)

        st.header("3. Tiền xử lý data")

        # Xử lý dữ liệu object
        categorical_cols = data.select_dtypes(include=['object']).columns
        st.write("Biến Categorical:")
        st.write(categorical_cols)

        # Áp dụng One-Hot Encoding
        data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

        # Hiển thị thông tin dữ liệu thiếu
        st.write("Giá trị bị null:")
        st.write(data.isnull().sum())

        # Chọn phương pháp xử lý dữ liệu thiếu
        missing_value_option = st.selectbox(
            "Chọn cách xử lý dữ liệu bị null?",
            ("Drop rows", "Fill with mean", "Fill with median", "Fill with mode")
        )

        # Xử lý dữ liệu thiếu
        if missing_value_option == "Drop rows":
            data = data.dropna()
        elif missing_value_option == "Fill with mean":
            data = data.fillna(data.mean())
        elif missing_value_option == "Fill with median":
            data = data.fillna(data.median())
        elif missing_value_option == "Fill with mode":
            data = data.fillna(data.mode().iloc[0])

        st.write("Data sau khi xử lý dữ liệu null:")
        st.write(data.head())

        st.header("4. Train Model")

        # Chọn cột mục tiêu
        target_column = st.selectbox("Chọn biến target", data.columns)

        # Chuẩn bị dữ liệu
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Chia dữ liệu thành tập huấn luyện và kiểm tra
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Xáo trộn dữ liệu huấn luyện
        X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train, random_state=42)

        # Chọn mô hình
        model_option = st.selectbox(
            "Chọn model để train",
            ("Linear Regression", "Logistic Regression",
             "K-Neighbors Classifier", "K-Neighbors Regressor",
             "Decision Tree Regressor", "Decision Tree Classifier",
             "Random Forest Classifier", "Random Forest Regressor",
             "Gradient Boosting Classifier", "Gradient Boosting Regressor")
        )

        if model_option == "Linear Regression":
            model = LinearRegression()
        elif model_option == "Logistic Regression":
            model = LogisticRegression()
        elif model_option == "K-Neighbors Classifier":
            model = KNeighborsClassifier(n_neighbors=5)
        elif model_option == "K-Neighbors Regressor":
            model = KNeighborsRegressor(n_neighbors=5)
        elif model_option == "Decision Tree Classifier":
            model = DecisionTreeClassifier(random_state=42)
        elif model_option == "Decision Tree Regressor":
            model = DecisionTreeRegressor(random_state=42)
        elif model_option == "Random Forest Classifier":
            model = RandomForestClassifier(max_depth=10)
        elif model_option == "Random Forest Regressor":
            model = RandomForestRegressor(max_depth=10)
        elif model_option == "Gradient Boosting Classifier":
            model = GradientBoostingClassifier(n_estimators=50)
        elif model_option == "Gradient Boosting Regressor":
            model = GradientBoostingRegressor(n_estimators=50)

        # Huấn luyện mô hình
        model.fit(X_train_shuffled, y_train_shuffled)

        # Dự đoán
        y_pred = model.predict(X_test)

        # Hiển thị kết quả huấn luyện
        st.write("Kết quả training")
        if model_option in ["Linear Regression", "K-Neighbors Regressor", "Decision Tree Regressor",
                            "Random Forest Regressor", "Gradient Boosting Regressor"]:
            st.write(f"Mean squared error: {mean_squared_error(y_test, y_pred):.2f}")
            st.write(f"R² score: {r2_score(y_test, y_pred):.2f}")

            # Vẽ biểu đồ
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
            ax.set_xlabel('Measured')
            ax.set_ylabel('Predicted')
            st.pyplot(fig)
        else:
            st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
            st.write("Classification Report:")
            st.text(classification_report(y_test, y_pred))
            st.write("Confusion Matrix:")
            st.write(confusion_matrix(y_test, y_pred))

        st.header("5. Dự đoán dữ liệu mới")

        # Sử dụng cột để nhập dữ liệu
        num_columns = len(X.columns)
        num_columns_per_row = 3  # Số lượng cột mỗi hàng
        num_rows = (num_columns + num_columns_per_row - 1) // num_columns_per_row

        input_data = {}
        col_index = 0

        for row in range(num_rows):
            cols = st.columns(num_columns_per_row)
            for col in cols:
                if col_index < num_columns:
                    column_name = X.columns[col_index]
                    input_data[column_name] = col.number_input(f"Input {column_name}", value=0)
                    col_index += 1

        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            st.write("Input Data")
            st.write(input_df)

            prediction = model.predict(input_df)
            st.write(f"Prediction: {prediction[0]}")


if __name__ == "__main__":
    st.set_page_config(page_title="Machine Learning Model", layout="wide")
    main()
