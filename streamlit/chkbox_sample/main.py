import streamlit as st
import pandas as pd
import shap
import plotly.figure_factory as ff
import matplotlib as plt
from sklearn.ensemble import RandomForestClassifier
import streamlit_shap

def main():
    st.set_page_config(
        page_title='チェックボックスとstate',
        layout='wide',
        page_icon = 'random'
        )
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title('checkbox')

    uploaded_file = st.file_uploader('データを読み込んで下さい。', type=['csv'], key = 'train_data')
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        
        tabs_set(data)

def tabs_set(data: pd.DataFrame):
    print("start tab set")
    tab_titles = ["データの確認", "解析", "前処理", "予測"]
    tabs = st.tabs(tab_titles)

    ss = st.session_state
    
    with tabs[0]:
        st.write('実験データの確認')
        st.write(data)

    with tabs[1]:
        if st.checkbox("plot"):
            if 'loaded' not in ss:
                st.write("start")
                group_labels = ["target"]
                # Create distplot with custom bin_size
                figs = []
                for i in range(0, len(data.columns)):
                    st.write(f"col:{data.columns[i]}")
                    fig = ff.create_distplot(
                            [data.iloc[:, i].values], group_labels, bin_size=0.25)

                    # Plot
                    st.plotly_chart(fig, use_container_width=True)
                    figs.append(fig)
                ss["loaded"] = figs
                
            else:
                st.write("loaded")
                figs = ss["loaded"]
                for fig in figs:
                    st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        if st.checkbox('SHAP解析', help='「実験点の提案」タブでモデルを構築してから'):
            st.write("start shap")
            if 'loaded_shap' not in ss:
                st.write("loaded_shap")
                clf = RandomForestClassifier(max_depth=2, random_state=0)
                X = data.iloc[:, :-1]
                y = data.iloc[:, -1]
                clf.fit(X, y)
                SHAP_explain(clf, X, X)
                ss["loaded_shap"] = True
            else:
                st.write("loaded")
                


def SHAP_explain(model, autoscaled_x, x):
    with st.spinner():
        st.write(autoscaled_x.head())
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(autoscaled_x)
        st.write(shap_values)
        waterfall_plot = shap.plots.waterfall(shap_values[0,:,1])
        streamlit_shap.st_shap(waterfall_plot, height=300)
        shap.summary_plot(shap_values, autoscaled_x, feature_names=x.columns)

if __name__ == '__main__':
    main()