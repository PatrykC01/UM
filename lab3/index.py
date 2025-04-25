import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

try:
    df = pd.read_csv('C:/Users/20pat/Downloads/UMData.csv')
    print(f"Wczytano dane. Kształt zbioru: {df.shape}")
except FileNotFoundError:
    print("Błąd: Plik 'myocardial infarction complications.csv' nie został znaleziony.")
    exit()

df.replace('?', np.nan, inplace=True)
print(f"\nZastąpiono '?' przez NaN.")

col_id = 'ID'
cols_outcomes_complications = [
    'FIBR_PREDS', 'PREDS_TAH', 'JELUD_TAH', 'FIBR_JELUD', 'A_V_BLOK',
    'OTEK_LANC', 'RAZRIV', 'DRESSLER', 'ZSN', 'REC_IM', 'P_IM_STEN',
    'LET_IS'
]
existing_outcomes = [col for col in cols_outcomes_complications if col in df.columns]
cols_to_exclude_from_features = [col_id] + existing_outcomes
cols_to_exclude_from_features = [col for col in cols_to_exclude_from_features if col in df.columns]

features_df = df.drop(columns=cols_to_exclude_from_features, errors='ignore')
print(f"\nKształt ramki danych z cechami (przed konwersją): {features_df.shape}")

numeric_cols = features_df.columns
for col in numeric_cols:
    features_df[col] = pd.to_numeric(features_df[col], errors='coerce')

print("\nPrzekonwertowano cechy na typ numeryczny (błędy na NaN).")
print(f"Liczba brakujących wartości (top 5):")
print(features_df.isnull().sum().sort_values(ascending=False).head())

preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

if features_df.shape[1] == 0:
     print("\nBŁĄD: Brak cech do analizy po wykluczeniu kolumn.")
     exit()

features_scaled = preprocessor.fit_transform(features_df)

features_scaled_df = pd.DataFrame(features_scaled, columns=features_df.columns, index=features_df.index)
print("\nCechy zostały zaimputowane (medianą) i przeskalowane (StandardScaler).")
print("Kształt przeskalowanych cech:", features_scaled_df.shape)

n_samples = features_scaled_df.shape[0]
if n_samples < 2:
    print("Za mało próbek danych (<2) do dalszej analizy.")
    exit()

inertia = []
silhouette_scores = []
k_range = range(2, 11)
valid_k_range_for_silhouette = []

for k in k_range:
    if k >= n_samples: 
        print(f"Pomijam k={k}, ponieważ jest >= niż liczba próbek ({n_samples})")
        continue
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) 
    kmeans.fit(features_scaled_df)
    inertia.append(kmeans.inertia_)
    try:
        score = silhouette_score(features_scaled_df, kmeans.labels_)
        silhouette_scores.append(score)
        valid_k_range_for_silhouette.append(k)
    except ValueError as e:
         print(f"Nie można obliczyć Silhouette Score dla k={k} na pełnym zbiorze. Błąd: {e}")

if inertia:
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    actual_k_tested_inertia = list(k_range)[:len(inertia)]
    plt.plot(actual_k_tested_inertia, inertia, marker='o', linestyle='-')
    plt.title('Metoda Łokcia (dane przed PCA)')
    plt.xlabel('Liczba klastrów (k)')
    plt.ylabel('Inercja (WSS)')
    plt.xticks(actual_k_tested_inertia)
    plt.grid(True)

if silhouette_scores:
    plt.subplot(1, 2, 2)
    plt.plot(valid_k_range_for_silhouette, silhouette_scores, marker='o', linestyle='-')
    plt.title('Wskaźnik Silhouette (dane przed PCA)')
    plt.xlabel('Liczba klastrów (k)')
    plt.ylabel('Średni Silhouette Score')
    plt.xticks(valid_k_range_for_silhouette)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # Wybór optymalnego k na podstawie Silhouette
    best_k_index = np.argmax(silhouette_scores)
    optimal_k = valid_k_range_for_silhouette[best_k_index]
    print(f"\nOptymalne k sugerowane przez Silhouette Score (przed PCA): {optimal_k} (score: {silhouette_scores[best_k_index]:.3f})")
else:
    optimal_k = 3 
    print(f"\nNie udało się obliczyć Silhouette Score. Używam domyślnego k={optimal_k}.")
    if inertia:
        plt.tight_layout()
        plt.show()

print(f"\n--- Krok 3: Klasteryzacja K-Means dla k={optimal_k} (przed PCA) ---")
kmeans_before_pca = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
labels_before_pca = kmeans_before_pca.fit_predict(features_scaled_df)
silhouette_before_pca = silhouette_score(features_scaled_df, labels_before_pca)
db_before_pca = davies_bouldin_score(features_scaled_df, labels_before_pca)
print(f"Wyniki K-Means przed PCA (k={optimal_k}):")
print(f"  - Silhouette Score: {silhouette_before_pca:.3f}")
print(f"  - Davies-Bouldin Index: {db_before_pca:.3f}")

print("\n--- Analiza PCA ---")
n_components_pca = features_scaled_df.shape[1]
if n_components_pca < 2:
    print("BŁĄD: Za mało cech (<2) do wykonania PCA.")
    exit()

pca = PCA(n_components=n_components_pca)
pca.fit(features_scaled_df)
explained_variance_ratio_cumulative = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, n_components_pca + 1), explained_variance_ratio_cumulative, marker='o', linestyle='--')
plt.title('Skumulowana wariancja wyjaśniona przez główne składowe (PCA)')
plt.xlabel('Liczba głównych składowych')
plt.ylabel('Skumulowany współczynnik wariancji wyjaśnionej')
plt.grid(True)
n_components_95 = np.argmax(explained_variance_ratio_cumulative >= 0.95) + 1
if n_components_95 > 0: 
    plt.axhline(y=0.95, color='r', linestyle=':', label=f'95% wariancji ({n_components_95} składowych)')
    plt.axvline(x=n_components_95, color='r', linestyle=':', label=f'{n_components_95} składowych')
else: 
    n_components_95 = n_components_pca 
    print("Nie osiągnięto 95% wariancji wyjaśnionej, rozważając użycie wszystkich składowych.")
plt.legend()
plt.show()
print(f"Liczba składowych potrzebna do wyjaśnienia >=95% wariancji: {n_components_95}")

# Redukcja do 2 wymiarów dla wizualizacji
pca_2d = PCA(n_components=2)
features_pca_2d = pca_2d.fit_transform(features_scaled_df)
pca_2d_df = pd.DataFrame(data=features_pca_2d,
                         columns=['Główna Składowa 1', 'Główna Składowa 2'],
                         index=features_scaled_df.index) 
print(f"\nWariancja wyjaśniona przez pierwsze 2 składowe: {pca_2d.explained_variance_ratio_.sum():.2%}")

pca_2d_df['Klaster_KMeans_Przed_PCA'] = labels_before_pca
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Główna Składowa 1', y='Główna Składowa 2', hue='Klaster_KMeans_Przed_PCA',
                data=pca_2d_df, palette='viridis', alpha=0.7, s=40, legend='full')
plt.title(f'Wizualizacja klastrów K-Means (k={optimal_k}, znalezionych PRZED PCA) na danych PCA 2D')
plt.xlabel('Główna Składowa 1')
plt.ylabel('Główna Składowa 2')
plt.grid(True)
plt.show()

pca_2d_df = pca_2d_df.drop(columns=['Klaster_KMeans_Przed_PCA'])

print(f"\n--- Klasteryzacja K-Means dla k={optimal_k} (po PCA do 2D) ---")
kmeans_after_pca = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
labels_after_pca = kmeans_after_pca.fit_predict(pca_2d_df) 
silhouette_after_pca = silhouette_score(pca_2d_df, labels_after_pca)
db_after_pca = davies_bouldin_score(pca_2d_df, labels_after_pca)
print(f"Wyniki K-Means po PCA (k={optimal_k}):")
print(f"  - Silhouette Score: {silhouette_after_pca:.3f}")
print(f"  - Davies-Bouldin Index: {db_after_pca:.3f}")

pca_2d_df['Klaster_KMeans_Po_PCA'] = labels_after_pca
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Główna Składowa 1', y='Główna Składowa 2', hue='Klaster_KMeans_Po_PCA',
                data=pca_2d_df, palette='viridis', alpha=0.7, s=40, legend='full')
centers = kmeans_after_pca.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.9, marker='X', label='Centroidy')
plt.title(f'Wyniki klasteryzacji K-Means (k={optimal_k}) na danych PCA 2D')
plt.xlabel('Główna Składowa 1')
plt.ylabel('Główna Składowa 2')
plt.legend()
plt.grid(True)
plt.show()

print("\nPorównanie jakości klasteryzacji K-Means przed i po PCA (2D):")
print(f"                          | Przed PCA | Po PCA (2D) |")
print(f"--------------------------|-----------|-------------|")
print(f"Silhouette Score          | {silhouette_before_pca:^9.3f} | {silhouette_after_pca:^11.3f} |")
print(f"Davies-Bouldin Index      | {db_before_pca:^9.3f} | {db_after_pca:^11.3f} |")


print("\n--- Klasteryzacja Hierarchiczna (na danych PCA 2D) ---")
data_for_hierarchical = pca_2d_df.drop(columns=['Klaster_KMeans_Po_PCA']) 

if data_for_hierarchical.shape[0] < 2:
    print("Za mało danych (<2 próbki) do klasteryzacji hierarchicznej.")
else:
    linkage_methods = ['ward', 'complete', 'average', 'single']
    plt.figure(figsize=(15, 10))
    plt.suptitle('Dendrogramy dla różnych metod łączenia (dane PCA 2D)', fontsize=16)
    valid_methods_count = 0
    linked_results = {} 

    for i, method in enumerate(linkage_methods):
        
        try:
            plt.subplot(2, 2, i + 1)
            sample_size_dendro = min(500, data_for_hierarchical.shape[0])
            if data_for_hierarchical.shape[0] > sample_size_dendro:
                 print(f"Rysowanie dendrogramu dla metody {method} na próbce {sample_size_dendro}...")
                 data_sample = data_for_hierarchical.sample(sample_size_dendro, random_state=42)
            else:
                 data_sample = data_for_hierarchical

            linked = linkage(data_sample, method=method)
            linked_results[method] = linkage(data_for_hierarchical, method=method) 
            dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=False, no_labels=True)
            plt.title(f'Metoda: {method}')
            plt.xlabel(f'Próbki (n={sample_size_dendro})')
            plt.ylabel('Odległość')
            valid_methods_count += 1
        except Exception as e:
            print(f"Nie można wygenerować dendrogramu dla metody '{method}'. Błąd: {e}")
            plt.subplot(2, 2, i + 1)
            plt.text(0.5, 0.5, f'Błąd dla metody\n{method}', ha='center', va='center')
            plt.title(f'Metoda: {method}')

    if valid_methods_count > 0:
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    else:
        plt.close()

    chosen_linkage_method = 'ward' 
    if chosen_linkage_method not in linked_results:
        chosen_linkage_method = 'average' 
        if chosen_linkage_method not in linked_results:
            chosen_linkage_method = list(linked_results.keys())[0] 

    print(f"\nWybrano metodę łączenia: '{chosen_linkage_method}' do wyodrębnienia klastrów.")

    linked_final = linked_results[chosen_linkage_method]

    labels_hier_k = fcluster(linked_final, t=optimal_k, criterion='maxclust')
    pca_2d_df[f'Klaster_Hier_{chosen_linkage_method}_k'] = labels_hier_k - 1

    try:
         distance_threshold = linked_final[-(optimal_k-1), 2] 
         distance_threshold_cut = distance_threshold * 0.95
         print(f"Przykładowy próg odległości do cięcia na {optimal_k} klastrów: {distance_threshold_cut:.2f}")
         labels_hier_dist = fcluster(linked_final, t=distance_threshold_cut, criterion='distance')
         pca_2d_df[f'Klaster_Hier_{chosen_linkage_method}_dist'] = labels_hier_dist - 1
         print(f"Wyodrębniono {len(np.unique(labels_hier_dist))} klastrów przy cięciu odległością.")

         # Wizualizacja dla obu cięć
         plt.figure(figsize=(18, 8))
         plt.subplot(1, 2, 1)
         sns.scatterplot(x='Główna Składowa 1', y='Główna Składowa 2', hue=f'Klaster_Hier_{chosen_linkage_method}_k',
                         data=pca_2d_df, palette='viridis', alpha=0.7, s=40, legend='full')
         plt.title(f'Hierarchiczna ({chosen_linkage_method}) - cięcie na k={optimal_k} klastrów (PCA 2D)')
         plt.xlabel('Główna Składowa 1')
         plt.ylabel('Główna Składowa 2')
         plt.grid(True)
         plt.legend()

         plt.subplot(1, 2, 2)
         sns.scatterplot(x='Główna Składowa 1', y='Główna Składowa 2', hue=f'Klaster_Hier_{chosen_linkage_method}_dist',
                         data=pca_2d_df, palette='viridis', alpha=0.7, s=40, legend='full')
         plt.title(f'Hierarchiczna ({chosen_linkage_method}) - cięcie odległością t={distance_threshold_cut:.2f} (PCA 2D)')
         plt.xlabel('Główna Składowa 1')
         plt.ylabel('Główna Składowa 2')
         plt.grid(True)
         plt.legend()
         plt.tight_layout()
         plt.show()

    except Exception as e:
        print(f"Nie można było wyodrębnić/wizualizować klastrów hierarchicznych: {e}")
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='Główna Składowa 1', y='Główna Składowa 2', hue=f'Klaster_Hier_{chosen_linkage_method}_k',
                        data=pca_2d_df, palette='viridis', alpha=0.7, s=40, legend='full')
        plt.title(f'Hierarchiczna ({chosen_linkage_method}) - cięcie na k={optimal_k} klastrów (PCA 2D)')
        plt.xlabel('Główna Składowa 1')
        plt.ylabel('Główna Składowa 2')
        plt.grid(True)
        plt.legend()
        plt.show()

    # Ocena jakości klasteryzacji Hierarchicznej (cięcie wg k)
    if optimal_k >= 2 and n_samples > optimal_k:
         try:
             silhouette_avg_hier = silhouette_score(data_for_hierarchical, labels_hier_k)
             db_score_hier = davies_bouldin_score(data_for_hierarchical, labels_hier_k)
             print(f"\nOcena klasteryzacji Hierarchicznej ({chosen_linkage_method}, cięcie na k={optimal_k}):")
             print(f"  - Średni Silhouette Score: {silhouette_avg_hier:.3f}")
             print(f"  - Davies-Bouldin Index: {db_score_hier:.3f}")
         except ValueError as e:
             print(f"\nNie można obliczyć wskaźników oceny dla klasteryzacji Hierarchicznej. Błąd: {e}")

    if 'Klaster_KMeans_Po_PCA' in pca_2d_df.columns and f'Klaster_Hier_{chosen_linkage_method}_k' in pca_2d_df.columns:
         comparison_df = pca_2d_df[['Klaster_KMeans_Po_PCA', f'Klaster_Hier_{chosen_linkage_method}_k']]
         print("\nPorównanie przypisań K-Means (po PCA) vs Hierarchiczny (cięcie na k):")
         contingency_table = pd.crosstab(comparison_df['Klaster_KMeans_Po_PCA'], comparison_df[f'Klaster_Hier_{chosen_linkage_method}_k'])
         print("\nTablica kontyngencji:")
         print(contingency_table)
    else:
         print("\nBrak wyników z obu metod klasteryzacji (po PCA) do porównania.")


print("\n--- Analiza charakterystyki klastrów (na podstawie K-Means po PCA) ---")
final_cluster_labels_col = 'Klaster_KMeans_Po_PCA' 

if final_cluster_labels_col in pca_2d_df.columns:
    final_cluster_labels = pca_2d_df[[final_cluster_labels_col]]
    df_analysis = df.join(final_cluster_labels, how='inner') 

    if final_cluster_labels_col not in df_analysis.columns:
        print("Błąd: Brak kolumny klastra w połączonych danych. Analiza niemożliwa.")
    else:
        print(f"\nAnaliza dla {optimal_k} klastrów znalezionych przez K-Means (po PCA):")
        analysis_results = {}
        
        if 'AGE' in df_analysis.columns:
            df_analysis['AGE'] = pd.to_numeric(df_analysis['AGE'], errors='coerce')
            analysis_results['Średni_Wiek'] = df_analysis.groupby(final_cluster_labels_col)['AGE'].mean()
       
        if 'SEX' in df_analysis.columns:
            df_analysis['SEX'] = pd.to_numeric(df_analysis['SEX'], errors='coerce')
            valid_sex = df_analysis[df_analysis['SEX'].isin([0, 1])]
            analysis_results['Procent_Kobiet'] = valid_sex.groupby(final_cluster_labels_col)['SEX'].mean() * 100
            
        if 'LET_IS' in df_analysis.columns:
            df_analysis['LET_IS'] = pd.to_numeric(df_analysis['LET_IS'], errors='coerce').fillna(-1)
            df_analysis['Zgon'] = df_analysis['LET_IS'].apply(lambda x: 1 if x > 0 else 0 if x == 0 else np.nan)
            analysis_results['Śmiertelność_%'] = df_analysis.groupby(final_cluster_labels_col)['Zgon'].mean() * 100
            analysis_results['Liczba_Pacjentów_w_Klasterze'] = df_analysis.groupby(final_cluster_labels_col)[final_cluster_labels_col].count()
        # Obrzęk Płuc
        if 'OTEK_LANC' in df_analysis.columns:
            df_analysis['OTEK_LANC'] = pd.to_numeric(df_analysis['OTEK_LANC'], errors='coerce')
            analysis_results['Obrzęk_Płuc_%'] = df_analysis.groupby(final_cluster_labels_col)['OTEK_LANC'].mean() * 100
        # Nadciśnienie (GB > 0)
        if 'GB' in df_analysis.columns:
            df_analysis['GB'] = pd.to_numeric(df_analysis['GB'], errors='coerce')
            df_analysis['Nadciśnienie'] = df_analysis['GB'].apply(lambda x: 1 if x > 0 else 0 if x == 0 else np.nan)
            analysis_results['Nadciśnienie_%'] = df_analysis.groupby(final_cluster_labels_col)['Nadciśnienie'].mean() * 100
        # Leukocyty
        if 'L_BLOOD' in df_analysis.columns:
            df_analysis['L_BLOOD'] = pd.to_numeric(df_analysis['L_BLOOD'], errors='coerce')
            analysis_results['Średnie_Leukocyty'] = df_analysis.groupby(final_cluster_labels_col)['L_BLOOD'].mean()

        if analysis_results:
             summary_table = pd.DataFrame(analysis_results)
             print("\n--- Podsumowanie Charakterystyki Klastrów (K-Means po PCA) ---")
             print(summary_table.round(2))
             summary_table.plot(kind='bar', subplots=True, figsize=(14, 12), layout=(-1, 3), legend=False,
                                title='Charakterystyka Klastrów K-Means (po PCA)')
             plt.tight_layout(rect=[0, 0.03, 1, 0.95])
             plt.show()
        else:
            print("Brak danych do analizy charakterystyki klastrów.")
else:
    print("\nBrak wyników klasteryzacji K-Means po PCA do analizy charakterystyki.")


print("\n--- Analiza zakończona ---")
