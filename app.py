import pickle
import shap

from shiny import App, ui, render, reactive
import pandas as pd

# Read csv and extract feature names
csv_path = "training_era.csv"
df = pd.read_csv(csv_path)
# Only retain required feature columns
feature_cols = [
    'age_at_brain_metastasis_diagnosis',
    'gender_m',
    'extracranial_disease_response_at_brain_metastasis_intervention_progressing',
    'number_of_brain_metastasis_n_miss',
    'number_of_brain_metastasis_1',
    'number_of_brain_metastasis_grt1',
    'pre_intervention_kps_grt80',
    'symptom_before_brain_metastasis_intervention_yes',
    'bm_location'
]

# Load explainer.pkl
with open('explainer.pkl', 'rb') as f:
    explainer = pickle.load(f)

# Observation data
observations = []

# Build input panel (update KPS label)
input_widgets = [
    ui.input_numeric('age_at_brain_metastasis_diagnosis', 'Age at brain metastasis diagnosis', 60, min=0, max=120),
    ui.input_select('gender_m', 'Gender', {'1': 'Male', '0': 'Female'}, selected='1'),
    ui.input_select('extracranial_disease_response_at_brain_metastasis_intervention_progressing', 'Extracranial disease status at brain metastasis intervention', {'1': 'Progressing', '0': 'Stable/responding'}, selected='0'),
    ui.input_select('number_of_brain_metastasis', 'Number of brain metastasis', {'1': '1', 'grt1': '>1'}, selected='1'),
    ui.input_select('bm_location', 'Brain metastasis location', {'infratentorial': 'Infratentorial', 'supratentorial': 'Supratentorial', 'both': 'Both'}, selected='supratentorial'),
    ui.input_select('pre_intervention_kps_grt80', 'Pre intervention KPS', {'1': '≥80', '0': '<80'}, selected='1'),
    ui.input_select('symptom_before_brain_metastasis_intervention_yes', 'Symptom before brain metastasis intervention', {'1': 'Yes', '0': 'No'}, selected='1'),
]

app_ui = ui.page_fluid(
    ui.h2(
        "Medical Prediction Demo (Real Feature Input)",
        style="text-align:center; font-weight:bold; margin-bottom:18px; letter-spacing:1px; font-size:2rem;"
    ),
    ui.layout_columns(
        ui.card(
            ui.h3("Patient Parameters", style="text-align:center; font-weight:bold; margin-bottom:14px; font-size:1.2rem;"),
            *input_widgets,
            ui.div(
                ui.input_action_button(
                    "predict", "Predict",
                    class_="btn-success",
                    style="width:48%;height:32px;font-size:15px;margin-right:4%;display:flex;align-items:center;justify-content:center;"
                ),
                ui.input_action_button(
                    "reset", "Reset",
                    class_="btn-secondary",
                    style="width:48%;height:32px;font-size:15px;display:flex;align-items:center;justify-content:center;"
                ),
                style="display:flex; justify-content:space-between; margin-top:12px;"
            ),
            style="padding:18px 18px 12px 18px; min-width:260px; max-width:500px; margin:auto; box-shadow:0 2px 12px rgba(0,0,0,0.08); border-radius:12px; background:white; margin-left:40px;"
        ),
        ui.div(
            ui.card(
                ui.h4("Waterfall Plot", style="font-weight:bold; margin-bottom:10px; font-size:1.1rem;"),
                ui.div(
                    ui.output_plot("shapplot"),
                    style="flex:1; display:flex; align-items:center; justify-content:center;"
                ),
                ui.div(
                    ui.h4("Force Plot", style="font-weight:bold; margin-top:18px; font-size:1.1rem;"),
                    ui.output_plot("forceplot"),
                    style="flex:4; display:flex; flex-direction:column; justify-content:center; height:0;"
                ),
                ui.div(
                    ui.h4("Survival Plot", style="font-weight:bold; margin-top:18px; font-size:1.1rem;"),
                    ui.output_plot("survival_curve"),
                    style="margin-top:10px;"
                ),
                style="padding:8px; min-width:600px; max-width:1600px; height:1200px; display:flex; flex-direction:column; gap:0; box-shadow:0 2px 8px rgba(0,0,0,0.07); border-radius:10px; background:white; margin-bottom:12px;"
            ),
            style="display:flex; flex-direction:column; gap:0; margin-left:-280px;"
        ),
        style="gap:24px; justify-content:center; align-items:flex-start; margin-bottom:20px;"
    )
)

def get_new_patient(input):
    training = pd.read_csv('training_era.csv')
    X_train = training.drop(["ID", "OS.time", "OS"], axis=1)
    new_patient = X_train.iloc[[0]].copy()

    if 'Age at BM Diagnosis' in new_patient.columns:
        new_patient['Age at BM Diagnosis'] = float(input.age_at_brain_metastasis_diagnosis())
    if 'Gender Male (vs Female)' in new_patient.columns:
        new_patient['Gender Male (vs Female)'] = int(input.gender_m())
    if 'Extracranial Disease Status at BM Intervention Progressing (vs Stable/Responding)' in new_patient.columns:
        new_patient['Extracranial Disease Status at BM Intervention Progressing (vs Stable/Responding)'] = int(input.extracranial_disease_response_at_brain_metastasis_intervention_progressing())
    if 'Number of BM >1 (vs 1)' in new_patient.columns:
        new_patient['Number of BM >1 (vs 1)'] = 1 if input.number_of_brain_metastasis() == 'grt1' else 0
    if 'Pre Intervention KPS >=80% (vs <80%)' in new_patient.columns:
        new_patient['Pre Intervention KPS >=80% (vs <80%)'] = int(input.pre_intervention_kps_grt80())
    if 'Symptoms Before BM Intervention Yes (vs No)' in new_patient.columns:
        new_patient['Symptoms Before BM Intervention Yes (vs No)'] = int(input.symptom_before_brain_metastasis_intervention_yes())
    # BM_Location赋值
    if 'BM Location Supratentorial (vs Infratentorial) ' in new_patient.columns:
        new_patient['BM Location Supratentorial (vs Infratentorial) '] = int(input.bm_location() == 'supratentorial')
    if 'BM Location Both (vs Infratentorial)' in new_patient.columns:
        new_patient['BM Location Both (vs Infratentorial)'] = int(input.bm_location() == 'both')
    # 其他特征保持模板默认值
    return new_patient

# 新增：记录是否已点击 Predict
predict_clicked = reactive.Value(False)

def server(input, output, session):
    @reactive.Calc
    def current_obs():
        return {
            'age_at_brain_metastasis_diagnosis': input.age_at_brain_metastasis_diagnosis(),
            'gender_m': input.gender_m(),
            'extracranial_disease_response_at_brain_metastasis_intervention_progressing': input.extracranial_disease_response_at_brain_metastasis_intervention_progressing(),
            'number_of_brain_metastasis': input.number_of_brain_metastasis(),
            'bm_location': input.bm_location(),
            'pre_intervention_kps_grt80': input.pre_intervention_kps_grt80(),
            'symptom_before_brain_metastasis_intervention_yes': input.symptom_before_brain_metastasis_intervention_yes(),
        }

    @reactive.Effect
    @reactive.event(input.predict)
    def add_observation():
        print("Predict button clicked!")
        print("current_obs:", current_obs())
        obs = current_obs()
        observations.append(obs)
        print("observations:", observations)
        predict_clicked.set(True)

    @reactive.Effect
    @reactive.event(input.reset)
    def reset_inputs():
        session.send_input_message('age_at_brain_metastasis_diagnosis', 60)
        session.send_input_message('gender_m', '1')
        session.send_input_message('extracranial_disease_response_at_brain_metastasis_intervention_progressing', '0')
        session.send_input_message('number_of_brain_metastasis', '1')
        session.send_input_message('bm_location', 'supratentorial')
        session.send_input_message('pre_intervention_kps_grt80', '1')
        session.send_input_message('symptom_before_brain_metastasis_intervention_yes', '1')
        predict_clicked.set(False)

    @reactive.Effect
    @reactive.event(
        input.age_at_brain_metastasis_diagnosis,
        input.gender_m,
        input.extracranial_disease_response_at_brain_metastasis_intervention_progressing,
        input.number_of_brain_metastasis,
        input.bm_location,
        input.pre_intervention_kps_grt80,
        input.symptom_before_brain_metastasis_intervention_yes,
    )
    def any_input_changed():
        predict_clicked.set(False)

    @output
    @render.plot
    def shapplot():
        print("shapplot called")
        import matplotlib.pyplot as plt
        if not predict_clicked.get():
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, 'Please click Predict', ha='center', va='center')
            ax.axis('off')
            return fig
        new_patient = get_new_patient(input)
        try:
            shap_values_new_patient = explainer(new_patient)
            sv = shap_values_new_patient[0]
            keep_idxs = [i for i, name in enumerate(sv.feature_names) if "n_miss" not in name]
            def format_feature_name(name):
                name = name.replace('_', ' ')
                if len(name) > 25:
                    words = name.split()
                    mid = len(words) // 2
                    return ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])
                return name
            new_names = [format_feature_name(sv.feature_names[i]) for i in keep_idxs]
            new_sv = shap.Explanation(
                values=sv.values[keep_idxs],
                base_values=sv.base_values,
                data=sv.data[keep_idxs],
                feature_names=[sv.feature_names[i] for i in keep_idxs]
            )
            new_sv.feature_names = new_names
            plt.figure(figsize=(20, 14))
            fig = shap.plots.waterfall(new_sv, show=False)
            ax = plt.gca()
            ax.tick_params(axis='y', labelsize=7, pad=20)
            ax.tick_params(axis='x', labelsize=10)
            plt.subplots_adjust(left=0.45, right=0.95, top=0.9, bottom=0.1)
            for label in ax.get_yticklabels():
                label.set_horizontalalignment('right')
                label.set_fontsize(7)
                label.set_fontweight('normal')
            return fig
        except Exception as e:
            import traceback
            traceback.print_exc()
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"SHAP error: {e}", ha='center', va='center')
            ax.axis('off')
            return fig

    @output
    @render.plot
    def forceplot():
        print("forceplot called")
        import matplotlib.pyplot as plt
        if not predict_clicked.get():
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, 'Please click Predict', ha='center', va='center')
            ax.axis('off')
            return fig
        new_patient = get_new_patient(input)
        try:
            shap_values_new_patient = explainer(new_patient)
            sv = shap_values_new_patient[0]
            keep_idxs = [i for i, name in enumerate(sv.feature_names) if "n_miss" not in name]
            def format_feature_name(name):
                formatted = name.replace('_', ' ')
                if len(formatted) > 30:
                    words = formatted.split()
                    if len(words) > 3:
                        mid = len(words) // 2
                        return ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])
                    elif len(words) == 3:
                        return words[0] + ' ' + words[1] + '\n' + words[2]
                    else:
                        if len(formatted) > 35:
                            mid = len(formatted) // 2
                            return formatted[:mid] + '\n' + formatted[mid:]
                return formatted
            new_sv = shap.Explanation(
                values=sv.values[keep_idxs],
                base_values=sv.base_values,
                data=sv.data[keep_idxs],
                feature_names=[format_feature_name(sv.feature_names[i]) for i in keep_idxs]
            )
            plt.figure(figsize=(24, 16))
            fig = shap.plots.force(new_sv, matplotlib=True, show=False)
            ax = plt.gca()
            texts = []
            for child in ax.get_children():
                if hasattr(child, 'get_text'):
                    texts.append(child)
            text_positions = []
            for i, text in enumerate(texts):
                if hasattr(text, 'get_position') and text.get_text():
                    text.set_fontsize(7)
                    pos = text.get_position()
                    text_positions.append((text, pos))
            sorted_texts = sorted(text_positions, key=lambda x: x[1][0])
            for i, (text, pos) in enumerate(sorted_texts):
                if i > 0:
                    prev_text, prev_pos = sorted_texts[i-1]
                    if abs(pos[0] - prev_pos[0]) < 2.5:
                        new_y = pos[1] + (0.05 if i % 2 == 0 else -0.05)
                        text.set_position((pos[0], new_y))
            plt.subplots_adjust(left=0.05, right=0.98, top=0.85, bottom=0.15)
            ax.set_ylim(ax.get_ylim()[0] - 0.1, ax.get_ylim()[1] + 0.1)
            return fig
        except Exception as e:
            import traceback
            traceback.print_exc()
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"SHAP error: {e}", ha='center', va='center')
            ax.axis('off')
            return fig

    @output
    @render.plot
    def survival_curve():
        print("survival_curve called")
        import matplotlib.pyplot as plt
        if not predict_clicked.get():
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, 'Please click Predict', ha='center', va='center')
            ax.axis('off')
            return fig
        new_patient = get_new_patient(input)
        try:
            import pickle
            with open('rsf_model.pkl', 'rb') as f:
                rsf = pickle.load(f)
            surv = rsf.predict_survival_function(new_patient, return_array=True)
            plt.figure(figsize=(7, 5))
            years = rsf.unique_times_
            plt.step(years, surv[0], where="post")
            plt.ylabel("Survival probability")
            plt.xlabel("Time (years)")
            plt.xlim(0, 12)
            plt.grid(True)
            return plt.gcf()
        except Exception as e:
            import traceback
            traceback.print_exc()
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Survival curve error: {e}", ha='center', va='center')
            ax.axis('off')
            return fig

app = App(app_ui, server)