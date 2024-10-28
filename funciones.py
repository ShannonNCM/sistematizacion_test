# funciones utilizadas para la sistematización de informacion

#librerias necesarias
import spacy
import pandas as pd
import numpy as np
import string
from collections import Counter
import matplotlib.pyplot as plt

# modulo NLP en español
nlp = spacy.load('es_core_news_md')


# se define un diccionario con las palabras clave que se quieren buscar dentro del texto que se esta analizando
dictionary = {
    'Datos' : ['dato'],
    'Agenda digital' : ['cambio de paradigma'],
    'Capacidades digitales' : ['capacitación', 'alfabetización', 'educación', 'formación', 'capacitar', 'educativo', 'conocimiento', 'preparación', 'taller'],
    'Gobernanza' : ['gobernanza', 'gobierno', 'confianza'],
    'Brecha digital' : ['acceso a todas', 'brecha digital', 'implementar programas', 'brecha'],
    'Marco legal' : ['ley', 'legal', 'normar', 'legalizar'],
    'Interoperabilidad' : [],
    'Identidad digital' : [],
    'Voluntad política' : [],
    'Presupuesto' : [],
    'Infraestructura publica' : ['infraestructura', 'acceso a la electricidad', 'equipo tecnológico', 'acceso a internet', 'internet a nivel nacional', 'acceso al internet'],
    'Pasarela de pagos' : [],
    'Simplificar tramites' : ['simplificación'],
    'Actualizar guatecompras' : [],
    'Digitalizacion de la informacion' : ['digitalizar', 'sistema'],
    'Difusion del eGob' : [],
    'Colaboracion intersectorial' : ['mesa de diálogo', 'alianza', 'interinstitucional', 'en conjunto'],
    'Intercambio de experiencias' : ['cooperación internacional'],
    'Participacion ciudadana' : ['libertad de expresión', 'auditoria social', 'llegar a todo', 'participación', 'participar', 'incluir'],
    'Estandarizacion tecnologica' : ['estandarización'],
    }

#funcion para extraer los datos
def extract_keywords(text, dict):
    text = text.translate(str.maketrans('','', string.punctuation)).lower()
    doc = nlp(text)
    extracted_concepts = set()
    
    lemmas = [token.lemma_ for token in doc]
    lemmas_text = ' '.join(lemmas)
    
    for termino, keywords in dict.items():
        for keyword in keywords:
            if keyword in lemmas_text:
                extracted_concepts.add(termino)
                
    return list(extracted_concepts)

#funcion para generar dataframes con las respuestas obtenidas
def expand_ans(x, col):
    terminos = set(termino for sublist in x[col] for termino in sublist)
    
    for termino in terminos:
        x[termino] = x[col].apply(lambda x: 1 if termino in x else 0)
    
    #esta parte agrega las columnas que faltan basado en los terminos del diccionario
    ditc_keys = dictionary.keys()
    
    for key in ditc_keys:
        if key not in x.columns:
            x[key] = 0
            
    other_cols = [col for col in x.columns if col not in ditc_keys]
    x = x[other_cols + list(ditc_keys)]
            
    return x

# esta funcion genera un dataframe con los resultados de la sistematizacion para cada pregunta
def results(df):
    grupo = df.groupby('Sector').sum().reset_index()
    grupo.loc['Total'] = grupo.sum()
    grupo.iloc[:, 1] = np.nan
    grupo.iloc[-1, 0] = np.nan
    
    return grupo

#funcion para graficar
def graf_rad(x):
    categories = x.columns
    values = x.iloc[-1]
    
    # se cierra la grafica repitiendo el primer valor
    values = np.append(values, values[0])
    categories = np.append(categories, categories[0])
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=True)[::-1]
    
    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, 'b', alpha=0.3)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories[:-1], fontsize=5)
    ax.set_theta_zero_location('N')
    
    return fig, ax