from datasets import load_dataset
import os
from transformers import pipeline


#classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k = 3)
#print(classifier("I hate this."))

from rdflib import Graph, Namespace, Literal, RDF, URIRef
from pyvis.network import Network
import json

#with open("genre_cache.json", "r", encoding="utf-8") as f:
#    data = json.load(f)

data = {
  "I Was Born About Ten Thousand Years Ago||Elvis Presley": "rock and roll",
  "Citadel||The Damned": "drama film",
  "Down the Drain||Down by Law": "drama film",
  "Hymn||Patti Smith": "Unknown",
  "Candy||LL Cool J": "pop music",
  "Little Birds||Dead to Fall": "indie rock",
  "Hannah||Sheila Nicholls": "Unknown",
  "Mental Slavery||Kreator": "thrash metal",
  "Playin' Dominoes and Shootin' Dice||Willie Nelson": "blues"
}

# Criar o grafo
g = Graph()

EX = Namespace("http://example.org/")
g.bind("", EX)

# Criar classes
g.add((EX.Music, RDF.type, EX.Class))
g.add((EX.Artist, RDF.type, EX.Class))
g.add((EX.Genre, RDF.type, EX.Class))

# Criar propriedades
g.add((EX.hasArtist, RDF.type, EX.Property))
g.add((EX.hasGenre, RDF.type, EX.Property))

# --------- ADICIONAR INDIVÍDUOS ---------
for key, genre in data.items():
    title, artist = key.split("||")

    # Criar URIs "limpos"
    title_uri = EX[title.replace(" ", "_").replace("/", "_").replace("\"", "").replace("'", "_")]
    artist_uri = EX[artist.replace(" ", "_").replace("/", "_").replace("\"", "")]
    genre_uri = EX[genre.replace(" ", "_").replace("/", "_")]

    # Música é indivíduo da classe Music
    g.add((title_uri, RDF.type, EX.Music))

    # Artista é indivíduo da classe Artist
    g.add((artist_uri, RDF.type, EX.Artist))

    # Género é indivíduo da classe Genre
    g.add((genre_uri, RDF.type, EX.Genre))

    # Relacionar a música com artista e género
    g.add((title_uri, EX.hasArtist, artist_uri))
    g.add((title_uri, EX.hasGenre, genre_uri))

g.serialize("musicas.ttl", format="turtle")

# Carregar o teu grafo RDFLib
g = Graph()
g.parse("musicas.ttl", format="turtle")

# --- preparar visualização PyVis com processamento mais limpo ---
net = Network(height="750px", width="100%", directed=True)
net.barnes_hut()  # layout melhor para grafos maiores

# iremos recolher tipos (Music / Artist / Genre) e depois construir nós/arestas sem triples rdf:type visíveis
class_uris = {EX.Music, EX.Artist, EX.Genre}
node_type = {}   # mapa: URI -> 'Music'|'Artist'|'Genre'|'Other'

# primeiro passar pelos triples para identificar rdf:type de instâncias
for s, p, o in g:
    if p == RDF.type and o in class_uris:
        # marca o tipo da instância (s é a instância, o é a classe)
        if o == EX.Music:
            node_type[s] = "Music"
        elif o == EX.Artist:
            node_type[s] = "Artist"
        elif o == EX.Genre:
            node_type[s] = "Genre"

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
    "Genre": "genre"
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