import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, accuracy_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from nltk.corpus import wordnet as wn
import random
from datetime import datetime
import os
import time
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

class DataPreprocessor:
    def __init__(self, unlabeled_path='active_learning/unlabeled_pool.csv'):
        self.unlabeled_path = unlabeled_path
        self.unlabeled_df = None
        self.train_df = None
        self.test_df = None

    def load_and_preprocess(self):
        self.unlabeled_df = pd.read_csv(self.unlabeled_path)
        self.unlabeled_df = self.unlabeled_df[~self.unlabeled_df['headword'].str.contains(r'\s')]
        self.unlabeled_df['headword'] = self.unlabeled_df['headword'].str.split('/').str[0]
        self.unlabeled_df = self.unlabeled_df.drop_duplicates(subset='headword')
        return self.unlabeled_df

    def create_train_test_splits(self, train_word_labels, test_word_labels):
        train_df = self.unlabeled_df[self.unlabeled_df['headword'].isin(train_word_labels.keys())].copy()
        train_df['is_complex'] = train_df['headword'].map(train_word_labels)
        self.train_df = train_df.drop_duplicates(subset='headword', keep='first').reset_index(drop=True)

        test_df = self.unlabeled_df[self.unlabeled_df['headword'].isin(test_word_labels.keys())].copy()
        test_df['is_complex'] = test_df['headword'].map(test_word_labels)
        self.test_df = test_df.drop_duplicates(subset='headword', keep='first').reset_index(drop=True)

        return self.train_df, self.test_df
    
    def create_clusters(self,n,unlabeled_pool,features):
         kmeans = KMeans(n_clusters=n, random_state=42, n_init='auto')
         features = unlabeled_pool[['freq', 'len']].values 
         cluster_labels = kmeans.fit_predict(features)
         unlabeled_df = unlabeled_pool.copy()
         unlabeled_df['k_cluster'] = cluster_labels
         return unlabeled_df

    

class FeatureExtractor:
    def __init__(self, features=['freq', 'len', 'wordnet_depth']):
        self.features = features
        self.scaler = StandardScaler()

    def calculate_wordnet_depth(self, word):
        synsets = wn.synsets(word)
        return max([len(hyp_path) for s in synsets for hyp_path in s.hypernym_paths()] or [0])

    def extract_features(self, df):
        if 'wordnet_depth' not in df.columns:
            df['wordnet_depth'] = df['headword'].apply(self.calculate_wordnet_depth)

        if 'cefr_score' not in df.columns and 'CEFR' in df.columns:
            cefr_map = {'A1': 1, 'A2': 2, 'B1': 3, 'B2': 4, 'C1': 5, 'C2': 6}
            df['cefr_score'] = df['CEFR'].map(cefr_map)

        X = df[self.features]
        if hasattr(self.scaler, 'scale_'):
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = self.scaler.fit_transform(X)
        return X_scaled

    def fit_scaler(self, X):
        self.scaler.fit(X)

class ActiveLearningModel:
    def __init__(self, api_key ,n =30,features=['freq', 'len', 'wordnet_depth']):
        self.model = LogisticRegression()
        self.feature_extractor = FeatureExtractor(features)
        self.train_word_labels = {}
        self.test_word_labels = {}
        self.api_key = api_key
        self.n=n
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
    

    def initiate_active_learning(self,  unlabeled_df, train_df, profile, classify_difficulty):
     for _ in range(self.n):
        unlabeled_scaled = self.feature_extractor.extract_features(unlabeled_df)

        # Predict probabilities + uncertainties
        probas = self.predict_proba(unlabeled_scaled)
        uncertainties = -np.sum(probas * np.log(probas + 1e-10), axis=1)
        class_ratio = train_df['is_complex'].mean()

        # Choose most informative sample
        selected_idx = np.argmax(probas[:, 1]) if class_ratio < 0.5 else np.argsort(uncertainties)[-1]
        selected_index = unlabeled_df.index[selected_idx]
        selected_sample = unlabeled_df.loc[[selected_index]].copy()

        # Label the selected word
        word = selected_sample['headword'].iloc[0]
        label =  classify_difficulty(profile,word,self.api_key)

        # Get distances within the same cluster
        cluster_id = selected_sample['k_cluster'].iloc[0]
        same_cluster_mask = unlabeled_df['k_cluster'] == cluster_id
        same_cluster_features = unlabeled_scaled[same_cluster_mask.to_numpy()]
        same_cluster_df = unlabeled_df[same_cluster_mask]

        # Map selected_idx (global) to cluster-local index
        local_idx = list(same_cluster_df.index).index(selected_index)

        selected_vector = same_cluster_features[local_idx].reshape(1, -1)
        distances = pairwise_distances(selected_vector, same_cluster_features)[0]
        closest_local_indices = np.argsort(distances)[:50]  # Choose 30 closest in the cluster
        closest_global_indices = same_cluster_df.iloc[closest_local_indices].index

        # Label the batch
        labeled_batch = unlabeled_df.loc[closest_global_indices].copy()
        labeled_batch['is_complex'] = label

        # Update pools
        unlabeled_df = unlabeled_df.drop(index=closest_global_indices).reset_index(drop=True)
        train_df = pd.concat([train_df, labeled_batch], ignore_index=True)


        # Retrain on updated train set
        X_train_scaled = self.feature_extractor.extract_features(train_df)
        self.train(X_train_scaled, train_df['is_complex'])

     return unlabeled_df, train_df
    

    def frequency_predictor(self, test_df, freq_threshold=4):
        return (test_df['freq'] <= freq_threshold).astype(int).tolist()

class Evaluator:
    def __init__(self):
        self.metrics_log = pd.DataFrame(columns=[
            'timestamp', 'accuracy', 'f1_score', 'recall',
            'precision', 'kappa', 'confusion_matrix', 'threshold_used'
        ])
       

    def evaluate(self, model, X_test, y_test,profile,frequency_threshold,y_freq_pred=None):
        y_pred = model.predict(X_test)

        metrics = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'kappa': cohen_kappa_score(y_test, y_pred),
            'confusion_matrix': str(confusion_matrix(y_test, y_pred)),
            'threshold_used': y_test.mean()
        }

        if y_freq_pred is not None:
            freq_metrics = {
                'accuracy': accuracy_score(y_test, y_freq_pred),
                'f1_score': f1_score(y_test, y_freq_pred),
                'recall': recall_score(y_test, y_freq_pred),
                'precision': precision_score(y_test, y_freq_pred),
                'kappa': cohen_kappa_score(y_test, y_freq_pred),
                'confusion_matrix': str(confusion_matrix(y_test, y_freq_pred)),
                'frequency_threshold':frequency_threshold
            }

            combined_metrics = {
                'timestamp': metrics['timestamp'],
                'accuracy': (metrics['accuracy'], freq_metrics['accuracy']),
                'f1_score': (metrics['f1_score'], freq_metrics['f1_score']),
                'recall': (metrics['recall'], freq_metrics['recall']),
                'precision': (metrics['precision'], freq_metrics['precision']),
                'kappa': (metrics['kappa'], freq_metrics['kappa']),
                'confusion_matrix': (metrics['confusion_matrix'], freq_metrics['confusion_matrix']),
                'threshold_used': metrics['threshold_used'],
                    'native_language': profile.get('native_language'),
                    'education_level': profile.get('education_level'),
                    'age_group': str(profile.get('age_group')),
                    'english_level': profile.get('english_level'),
                    'job_category': profile.get('job_category'),
                    'reading_domain':profile.get('reading_domain')
            }


            self.metrics_log = pd.concat([self.metrics_log, pd.DataFrame([combined_metrics])], ignore_index=True)
            return combined_metrics
        else:
            self.metrics_log = pd.concat([self.metrics_log, pd.DataFrame([metrics])], ignore_index=True)
            return metrics

    def save_metrics(self, filename='metrics_log_new.csv'):
     if os.path.exists(filename):
        # Append without writing header
        self.metrics_log.to_csv(filename, mode='a', index=False, header=False)
     else:
        # File doesn't exist, write normally with header
        self.metrics_log.to_csv(filename, index=False)
     print("Metrics successfully logged:")
     print(self.metrics_log.tail())

class UserInterface:
    def __init__(self,api_key):
        self.train_word_labels = {}
        self.test_word_labels = {}
        self.api_key =api_key

    def collect_train_labels(self, unlabeled_pool, profile, classify_difficulty, num_samples=30,):
        cefr_map = {'A1': 1, 'A2': 2, 'B1': 3, 'B2': 4, 'C1': 5, 'C2': 6}
        unlabeled_pool = unlabeled_pool[unlabeled_pool['CEFR'].isin(cefr_map)]
        unlabeled_pool['cefr_score'] = unlabeled_pool['CEFR'].map(cefr_map)

        while len(self.train_word_labels) < num_samples and not unlabeled_pool.empty:
            num_labeled = len(self.train_word_labels)
            num_simple = sum(1 for lbl in self.train_word_labels.values() if lbl == 0)
            simple_ratio = num_simple / (num_labeled + 1e-6)

            if simple_ratio < 0.5:
                sample_pool = unlabeled_pool.sample(n=10, random_state=random.randint(0, 10000))
            else:
                top_half = unlabeled_pool.sort_values(by='cefr_score', ascending=False)
                sample_pool = top_half.head(30).sample(n=10, random_state=random.randint(0, 10000))

            row = sample_pool.sample(1, random_state=random.randint(0, 10000)).iloc[0]
            word = row['headword']

           
            
            label = classify_difficulty(profile,word,self.api_key)
           
            self.train_word_labels[word] = label
            unlabeled_pool = unlabeled_pool[unlabeled_pool['headword'] != word]
                   

        print("\n✅ Done! You labeled 30 words.")
        return self.train_word_labels , unlabeled_pool

    def collect_test_labels(self, unlabeled_pool, profile, classify_difficulty, num_samples=100,):
        cefr_map = {'A1': 1, 'A2': 2, 'B1': 3, 'B2': 4, 'C1': 5, 'C2': 6}
        unlabeled_pool = unlabeled_pool[unlabeled_pool['CEFR'].isin(cefr_map)]
        unlabeled_pool['cefr_score'] = unlabeled_pool['CEFR'].map(cefr_map)

        while len(self.test_word_labels) < num_samples and not unlabeled_pool.empty:
            num_labeled = len(self.test_word_labels)
            num_simple = sum(1 for lbl in self.test_word_labels.values() if lbl == 0)
            simple_ratio = num_simple / (num_labeled + 1e-6)

            if simple_ratio < 0.5:
                sample_pool = unlabeled_pool.sample(n=10, random_state=random.randint(0, 10000))
            else:
                top_half = unlabeled_pool.sort_values(by='cefr_score', ascending=False)
                sample_pool = top_half.head(30).sample(n=10, random_state=random.randint(0, 10000))

            row = sample_pool.sample(1, random_state=random.randint(0, 10000)).iloc[0]
            word = row['headword']

            label = classify_difficulty(profile,word,self.api_key)
            
            self.test_word_labels[word] = label
            unlabeled_pool = unlabeled_pool[unlabeled_pool['headword'] != word]
                

        print("\n✅ Done! You labeled 50 words.")
        return self.test_word_labels, unlabeled_pool

# === Main Execution ===



from  simulate_user import classify_word_complexity
import os
from dotenv import load_dotenv



class UserSimulation:
    def __init__(self, preprocessor, ui, feature_extractor, model, evaluator, api_key):
        self.preprocessor = preprocessor
        self.ui = ui
        self.feature_extractor = feature_extractor
        self.model = model
        self.evaluator = evaluator
        self.api_key = api_key

    def simulate_user(self, profile, max_retries=3, retry_delay=5):
        for attempt in range(max_retries):
            try:
                # Load and preprocess
                unlabeled_df = self.preprocessor.load_and_preprocess()
                
                # Extract features (returns numpy array)
                X_features = self.feature_extractor.extract_features(unlabeled_df)
                
                # Cluster using the extracted features
                unlabeled_df = self.preprocessor.create_clusters(5, unlabeled_df, X_features)

                # Label collection with retry handling
                train_word_labels, unlabeled_pool = self.ui.collect_train_labels(
                    unlabeled_df.copy(),
                    profile=profile,
                    classify_difficulty=classify_word_complexity
                )
                
                test_word_labels, unlabeled_pool = self.ui.collect_test_labels(
                    unlabeled_pool.copy(),
                    profile=profile,
                    classify_difficulty=classify_word_complexity
                )
                
                # Rest of the simulation logic...
                train_df, test_df = self.preprocessor.create_train_test_splits(train_word_labels, test_word_labels)
                freq_threshold = train_df['freq'].mean()
                X_train = self.feature_extractor.extract_features(train_df)
                y_train = train_df['is_complex']
                self.model.train(X_train, y_train)

                unlabeled_df, train_df = self.model.initiate_active_learning(
                    unlabeled_df=unlabeled_pool,
                    train_df=train_df,
                    profile=profile,
                    classify_difficulty=classify_word_complexity
                )

                X_test = self.feature_extractor.extract_features(test_df)
                y_test = test_df['is_complex']
                y_freq = self.model.frequency_predictor(test_df, freq_threshold=freq_threshold)

                metrics = self.evaluator.evaluate(self.model, X_test, y_test, profile, freq_threshold, y_freq)
                self.evaluator.save_metrics()
                
                # If we get here, everything worked - break out of retry loop
                break

            except Exception as e:
                if "OpenAI" in str(e) or "API" in str(e):  # Catch API-related errors
                    if attempt < max_retries - 1:
                        print(f"⚠️ OpenAI API error (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        print(f"❌ Failed after {max_retries} attempts. Skipping profile: {profile}")
                        return None
                else:
                    # Re-raise non-API errors
                    raise










def main():
    load_dotenv()
    api_key = os.getenv("API_KEY")
    preprocessor = DataPreprocessor()
  
   



    import random
    import itertools

    native_languages = ['Spanish','Mandarin','Arabic','Russian','French','Italian','Turkish']
    last_completed_education_level= ['High School','Bachelors','Masters','Ph.D.']
    age_groups = [(18,25),(25,30),(30,40)]
    english_level = ['Beginner','Intermediate','Advanced']
    job_categories = [
        "Business & Office Work",
        "Technology & Engineering",
        "Healthcare",
        "Education & Research",
        "Service & Hospitality",
        "Creative & Communication",
        "Skilled Trades & Vocational",
        "Government & Legal"
    ]
    reading_domain=['Businness News','Fiction','Non Fiction', 'Social Media','Academic papers']
    
    keys = ['native_language', 'education_level', 'age_group', 'english_level', 'job_category','reading_domain']

    all_combinations = [
        dict(zip(keys, values))
        for values in itertools.product(
            native_languages,
            last_completed_education_level,
            age_groups,
            english_level,
            job_categories,
            reading_domain
        )
    ]

    advanced_profiles = [
    profile for profile in all_combinations
    if profile['english_level'] == 'Advanced'
]
    API_KEY =os.getenv("API_KEY")
    print(API_KEY)

    num_samples = 50 

    
    sampled_profiles = random.sample(all_combinations, k=min(num_samples, len(all_combinations)))


    for profile in sampled_profiles:
          time.sleep(1)
          ui = UserInterface(api_key=api_key)
          feature_extractor = FeatureExtractor()
          model = ActiveLearningModel(api_key,30)
          evaluator = Evaluator()

    

          userSimulator = UserSimulation(preprocessor=preprocessor,
                                   ui=ui,
                                   feature_extractor=feature_extractor,
                                   model=model,
                                   evaluator=evaluator,
                                   api_key=api_key
                                   )
          userSimulator.simulate_user(profile)
 

if __name__ == "__main__":
    main()