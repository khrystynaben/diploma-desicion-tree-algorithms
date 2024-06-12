import pandas as pd
import random

def clean_table(input_file, output_file):
    # Завантаження CSV файлу в DataFrame
    df = pd.read_csv(input_file)
    # Видалення дублікатів
    df_cleaned = df.drop_duplicates()

    # Видалення рядків, де відрізняється тільки один стовпчик
    unique_rows = []
    for index, row in df_cleaned.iterrows():
        row_values = tuple(row.values)
        if row_values not in unique_rows:
            unique_rows.append(row_values)

    df_final = pd.DataFrame(unique_rows, columns=df.columns)
     # Збереження зміненого DataFrame у новий CSV файл
    df_final.to_csv(output_file, index=False)


# Приклад використання:
data = pd.DataFrame({
    'Column1': [1, 2, 3, 4, 5, 1],
    'Column2': ['A', 'B', 'C', 'D', 'E', 'A'],
    'Column3': [True, False, True, True, False, True]
})

#------------------------------------------------------------
# -------видалити продубльовані рядочки----------------------
#------------------------------------------------------------

def remove_duplicate_rows(input_file, output_file):
    # Завантаження CSV файлу в DataFrame
    df = pd.read_csv(input_file)
    
    # Видалення повторюваних рядків
    df.drop_duplicates(inplace=True)
    
    # Збереження зміненого DataFrame у новий CSV файл
    df.to_csv(output_file, index=False)
'''
# Приклад використання функції
input_file = 'output3.csv'  # Ваш CSV файл
output_file = 'output4.csv'  # Файл для збереження без повторів
clean_table(input_file, output_file)
'''

#------------------------------------------------------------
# -------змінити назву файлу-----
#------------------------------------------------------------
def update_csv(file_path, column_number,new_name):
    # Завантажуємо дані з CSV файлу
    df = pd.read_csv(file_path)
    
    # Перевіряємо наявність потрібного стовпця
    if df.columns[0] not in df.columns:
        raise ValueError(f"Column {df.columns[0]} does not exist in the data.")
    
    print(df[df.columns[column_number]])
    
    # Змінюємо значення першого стовпця за умовою
    df[df.columns[column_number]] = df[df.columns[column_number]].map({' Approved': 1, ' Rejected': 0})
    
    print(df[df.columns[column_number]])
    # Зберігаємо зміни у новий CSV файл
    df.to_csv(new_name, index=False)
    print(f"Updated file saved as 'updated_'")

'''
# Виклик функції з шляхом до вашого CSV файлу
file_path = 'data/loan!/loan_approval_train.csv'  # замініть на шлях до вашого файлу
update_csv(file_path, 11, 'data/loan!/loan_approval_train333.csv')
'''



def remove_spaces_from_csv(input_file, output_file):
    # Завантаження CSV файлу
    df = pd.read_csv(input_file)
    
    # Видалення пробілів з назв стовпчиків
    df.columns = df.columns.str.replace(' ', '')

    # Видалення пробілів з значень
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = df[column].str.replace(' ', '')

    # Збереження очищених даних назад у CSV файл
    df.to_csv(output_file, index=False)


# Використання функції
input_file = 'data/loan!/loan_approval_train333.csv'
output_file = 'data/loan!/loan_approval_train_updated.csv'
remove_spaces_from_csv(input_file, output_file)




def merge_columns(file1, file2, output_file):
    # Завантажити файли
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # Об'єднати стовпці
    merged_df = pd.concat([df1, df2], axis=1)
    
    # Зберегти результат у новий файл
    merged_df.to_csv(output_file, index=False)
    print(f"Файл з об'єднаними стовпцями збережено як {output_file}")
'''
# Використання функції
file1 = 'data/aut_prediction/test3.csv'
file2 = 'data/aut_prediction/sample_submission3.csv'
output_file = 'data/aut_prediction/merge.csv'
merge_columns(file1, file2, output_file)
'''
#------------------------------------------------------------
# --------------пересунути стовбець у кінець----------------
#------------------------------------------------------------

def move_column_to_end(file_path, new_file_path, column_name):
    # Завантажуємо дані з CSV файлу
    df = pd.read_csv(file_path)
    
    # Перевіряємо наявність потрібного стовпця
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} does not exist in the data.")
    
    # Переміщуємо стовпець на останню позицію
    columns = list(df.columns)
    columns.remove(column_name)
    #columns.append(column_name)
    df = df[columns]
    
    # Зберігаємо зміни у новий CSV файл
    df.to_csv(new_file_path, index=False)
    print(f"file saved")
'''
# Виклик функції з шляхом до вашого CSV файлу та назвою стовпця
file_path = 'data/aut_prediction/train3.csv'  # замініть на шлях до вашого файлу
new_file_path = 'data/aut_prediction/train4.csv'
column_name = 'ID'  # замініть на назву стовпця, який потрібно перемістити
move_column_to_end(file_path, new_file_path, column_name )
'''
import pandas as pd

def delete_columns(file, columns_to_delete, output_file):
    # Завантажити файл
    df = pd.read_csv(file)
    
    # Видалити вказані стовпчики
    df = df.drop(columns=columns_to_delete)
    
    # Зберегти результат у новий файл
    df.to_csv(output_file, index=False)
    print(f"Файл з видаленими стовпчиками збережено як {output_file}")
'''
# Використання функції
file = 'data/loan!/loan_approval_dataset.csv'
#columns_to_delete = ['ethnicity', 'age', 'gender','jaundice','austim','contry_of_res','used_app_before', 'age_desc', 'result', 'relation']  # Вкажіть перелік стовпчиків для видалення
columns_to_delete = ['loan_id']
output_file = 'data/loan!/loan_approval_datase-new.csv'
delete_columns(file, columns_to_delete, output_file)
'''
#------------------------------------------------------------
#-------------- замінити знаки на коми--------------------
#------------------------------------------------------------
'''
input_file_path = 'data/divorce/test_divorce_data.csv'
output_file_path = 'transformed_file3.csv'

# Відкриваємо початковий файл і зчитуємо його вміст
with open(input_file_path, 'r', encoding='utf-8') as file:
    file_data = file.read()

# Заміна всіх ';' на ','
file_data = file_data.replace(';', ',')

# Записуємо змінені дані у новий файл
with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write(file_data)

print("File transformation complete.")
#------------------------------------------------------------
'''

#------------------------------------------------------------
# -------витягнути певну кількість елементів з файлу-----
#------------------------------------------------------------
def extract_random_rows(input_file, output_file, num_rows):
    # Читаємо CSV файл у DataFrame
    df = pd.read_csv(input_file)
    
    # Переконуємось, що кількість рядків не перевищує кількість рядків у файлі
    if num_rows > len(df):
        raise ValueError("The number of rows requested exceeds the total number of rows in the file.")
    
    # Вибираємо випадкові рядки
    random_indices = random.sample(range(len(df)), num_rows)
    random_rows = df.iloc[random_indices]
    
    # Записуємо випадкові рядки у новий CSV файл
    random_rows.to_csv(output_file, index=False)

'''
# Використання функції
input_file_path = 'data/satisfaction/re_test20k.csv'
output_file_path = 'data/satisfaction/test4000.csv'
number_of_rows = 4000  # Кількість випадкових рядків, які потрібно витягнути

extract_random_rows(input_file_path, output_file_path, number_of_rows)
'''

#------------------------------------------------------------
# -------порахувати кількість елементів в стовпцях-----------
#------------------------------------------------------------

def count_elements_in_columns(input_file):
    # Читаємо CSV файл у DataFrame
    df = pd.read_csv(input_file)
    
    # Створюємо словник для зберігання результатів
    column_counts = {}
    
    # Підраховуємо унікальні елементи в кожному стовпці
    for column in df.columns:
        column_counts[column] = df[column].value_counts().to_dict()
    
    return column_counts
'''
# Використання функції
input_file_path = 'data/satisfaction/re_general_01.csv'
column_counts = count_elements_in_columns(input_file_path)

# Виведення результатів
for column, counts in column_counts.items():
    print(f"Column '{column}':")
    for element, count in counts.items():
        print(f"  {element}: {count}")
'''

#------------------------------------------------------------
#-------------- розділення файлів на 2 ----------------------
#------------------------------------------------------------

def split_csv_randomly(input_file, output_file_1, output_file_2, size_file_1, size_file_2):
    # Зчитуємо CSV файл у DataFrame
    df = pd.read_csv(input_file)
    
    # Перемішуємо рядки DataFrame
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Перевіряємо, чи розміри файлів не перевищують загальну кількість рядків
    if size_file_1 + size_file_2 > len(df):
        raise ValueError("Задані розміри файлів перевищують загальну кількість рядків у файлі.")
    
    # Розділяємо DataFrame на два окремі DataFrame
    df_file_1 = df.iloc[:size_file_1]
    df_file_2 = df.iloc[size_file_1:size_file_1 + size_file_2]
    
    # Зберігаємо DataFrame у два окремі CSV файли
    df_file_1.to_csv(output_file_1, index=False)
    df_file_2.to_csv(output_file_2, index=False)

'''
# Використання функції
input_file_path = 'data/loan!/loan_approval_datase-new.csv'
output_file_path_1 = 'data/loan!/loan_approval_train.csv'
output_file_path_2 = 'data/loan!/loan_approval_test.csv'
size_file_1 = 3500
size_file_2 = 699

split_csv_randomly(input_file_path, output_file_path_1, output_file_path_2, size_file_1, size_file_2)
'''