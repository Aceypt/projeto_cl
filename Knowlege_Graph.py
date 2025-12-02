#from datasets import load_dataset
import os
from transformers import pipeline
import pandas as pd


#classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k = 3)
#data = pd.read_csv("lyrics_with_genre.csv")
#
#data['seq'] = data['seq'].str.replace("_x000D_", "", regex=False)
#
#dataHead = data.head()
#
#print(dataHead)
#x = dataHead.iloc[0]["seq"]
#print(classifier(x))
#print(x)

from rdflib import Graph, Namespace, RDF
from pyvis.network import Network
import ast

data = pd.read_csv("final.csv")

data = data.head(10)

# FUNÇÃO PARA LIMPAR URIs
def clean_uri(x):
    return x.replace(" ", "_").replace("/", "_").replace("\"", "").replace("'", "_")

# CRIAR GRAFO RDF
g = Graph()
EX = Namespace("http://example.org/")
g.bind("", EX)

# DEFINIR CLASSES
g.add((EX.Music, RDF.type, EX.Class))
g.add((EX.Artist, RDF.type, EX.Class))
g.add((EX.Genre, RDF.type, EX.Class))
g.add((EX.Emotion, RDF.type, EX.Class))

# DEFINIR PROPRIEDADES
g.add((EX.hasArtist, RDF.type, EX.Property))
g.add((EX.hasGenre, RDF.type, EX.Property))
g.add((EX.hasEmotion, RDF.type, EX.Property))

for _, row in data.iterrows():

    title_uri   = EX[clean_uri(row["song"])]
    artist_uri  = EX[clean_uri(row["artist"])]
    genre_uri   = EX[clean_uri(row["genre"])]

    # --- converter o texto '{'sadness':0.83}' num dicionário ---
    emotions_dict = ast.literal_eval(row["predicted_emotions"])

    # --- criar URIs para todas as emoções ---
    emotion_uris = [EX[clean_uri(em)] for em in emotions_dict.keys()]

    # tipos das instâncias
    g.add((title_uri, RDF.type, EX.Music))
    g.add((artist_uri, RDF.type, EX.Artist))
    g.add((genre_uri, RDF.type, EX.Genre))

    for em_uri in emotion_uris:
        g.add((em_uri, RDF.type, EX.Emotion))

    # relações
    g.add((title_uri, EX.hasArtist, artist_uri))
    g.add((title_uri, EX.hasGenre, genre_uri))

    # ligar música → todas as emoções
    for em_uri in emotion_uris:
        g.add((title_uri, EX.hasEmotion, em_uri))

g.serialize("musicas.ttl", format="turtle")

g = Graph()
g.parse("musicas.ttl", format="turtle")

# --- preparar visualização PyVis com processamento mais limpo ---
net = Network(height="750px", width="100%", directed=True)
net.barnes_hut()  # layout melhor para grafos maiores

# iremos recolher tipos (Music / Artist / Genre) e depois construir nós/arestas sem triples rdf:type visíveis
class_uris = {EX.Music, EX.Artist, EX.Genre, EX.Emotion}
node_type = {}   # mapa: URI -> 'Music'|'Artist'|'Genre'|'Other'

# Identificar tipos
for s, p, o in g:
    if p == RDF.type and o in class_uris:
        if o == EX.Music:
            node_type[s] = "Music"
        elif o == EX.Artist:
            node_type[s] = "Artist"
        elif o == EX.Genre:
            node_type[s] = "Genre"
        elif o == EX.Emotion:
            node_type[s] = "Emotion"

# depois criar nós e arestas (ignorando triples que apenas declaram as classes em si)
seen_nodes = set()

def pretty_label(uri):
    """Gera label legível: tenta qname, fallback para o último segmento do URIRef"""
    try:
        return g.qname(uri)
    except Exception:
        s = str(uri)
        return s.split("/")[-1].split("#")[-1]

# cores/grupos para PyVis (o "group" facilita legenda/estética)
group_map = {
    "Music": "music",
    "Artist": "artist",
    "Genre": "genre",
    "Emotion": "emotion"
}

# Adicionar nós e arestas: para cada triple, se for rdf:type (instância->classe) já processado -> ignorar visualmente.
for s, p, o in g:
    # Só mostrar indivíduos, não classes
    if o in (EX.Music, EX.Artist, EX.Genre):
        continue
    # ignorar declarações do próprio esquema (ex.: EX.Music rdf:type EX.Class) e rdf:type ligações já processadas
    if s in {EX.Music, EX.Artist, EX.Genre}:
        continue
    if p == RDF.type and o in class_uris:
        # Não criar aresta rdf:type visível — apenas asseguramos node_type acima
        continue

    # garantir nós s e o com labels legíveis
    if s not in seen_nodes:
        lbl = pretty_label(s)
        grp = group_map.get(node_type.get(s, "Other"), "other")
        net.add_node(str(s), label=lbl, title=str(s), group=grp)
        seen_nodes.add(s)
    if o not in seen_nodes:
        lbl = pretty_label(o)
        grp = group_map.get(node_type.get(o, "Other"), "other")
        net.add_node(str(o), label=lbl, title=str(o), group=grp)
        seen_nodes.add(o)

    # adicionar aresta com rótulo do predicado (localname)
    pred_label = pretty_label(p)
    net.add_edge(str(s), str(o), label=pred_label, title=pred_label)


# PyVis aplica cores automaticamente por group.
net.set_options("""
var options = {
  "nodes": {
    "font": {"size": 14}
  },
  "edges": {
    "arrows": {"to": {"enabled": true}},
    "font": {"align": "top"}
  },
  "physics": {
    "stabilization": { "enabled": true }
  }
}
""")

net.write_html("grafico_interativo_limpo.html")