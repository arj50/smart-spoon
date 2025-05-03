import pandas as pd
import matplotlib.pyplot as plt

# Load the survey data
file_path = file_path = 'data/survey_data.csv'
  # ensure this file is in the same directory

def load_and_clean(csv_path):
    df = pd.read_csv(csv_path)

    # Rename columns for consistency
    df = df.rename(columns={
        df.columns[1]: 'Age',
        df.columns[2]: 'Gender',
        df.columns[3]: 'Low_Sodium_Diet',
        df.columns[4]: 'Medical_Condition',
        df.columns[5]: 'Eat_Out_Freq',
        df.columns[6]: 'Low_Sodium_Satisfaction',
        df.columns[7]: 'Added_Salt',
        df.columns[8]: 'Aware_Tech',
        df.columns[9]: 'Interest_Device',
        df.columns[10]: 'Importance',
        df.columns[11]: 'Expected_Features',
        df.columns[12]: 'Purchase_Intent',
        df.columns[13]: 'Concerns',
        df.columns[14]: 'Suggestions',
        df.columns[15]: 'Salt_Dal',
        df.columns[16]: 'Salt_Sambar',
        df.columns[17]: 'Salt_Biryani',
        df.columns[18]: 'Salt_Curries',
        df.columns[19]: 'Salt_Snacks',
        df.columns[20]: 'Salt_Roti',
        df.columns[21]: 'Salt_Pickles',
        df.columns[22]: 'Salt_Content'
    })

    # Clean Age column
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df = df.dropna(subset=['Age'])
    df['Age'] = df['Age'].astype(int)
    return df


def plot_distribution(df):
    # 1. Age distribution
    plt.figure(figsize=(6,4))
    plt.hist(df['Age'], bins=10, edgecolor='black')
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

    # 2. Low sodium diet followers
    counts = df['Low_Sodium_Diet'].value_counts()
    plt.figure(figsize=(4,4))
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Following Low-Sodium Diet')
    plt.tight_layout()
    plt.show()

    # 3. Awareness of taste tech
    counts = df['Aware_Tech'].value_counts()
    plt.figure(figsize=(4,4))
    plt.bar(counts.index, counts.values)
    plt.title('Awareness of Taste-Enhancing Technology')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 4. Interest in device
    counts = df['Interest_Device'].value_counts()
    plt.figure(figsize=(4,4))
    plt.bar(counts.index, counts.values)
    plt.title('Interest in Enhanced-Taste Device')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 5. Salt content satisfaction
    counts = df['Salt_Content'].value_counts()
    plt.figure(figsize=(4,4))
    plt.bar(counts.index, counts.values)
    plt.title('Salt Content Satisfaction')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    df = load_and_clean(file_path)
    print("Data loaded and cleaned. Total respondents:", len(df))
    plot_distribution(df)

