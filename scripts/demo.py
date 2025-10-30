from src.search_engine import SearchEngine
from config import MODEL_NAME

DOCUMENTS = [
  "Elephants are the largest living terrestrial animals. Three living species are currently recognised: the African bush elephant (Loxodonta africana), the African forest elephant (L. cyclotis), and the Asian elephant (Elephas maximus). They are the only surviving members of the family Elephantidae and the order Proboscidea.",
  "A dolphin is a common name used for some of the aquatic mammals in the cetacean clade Odontoceti, the toothed whales. Dolphins belong to the families Delphinidae (the oceanic dolphins), along with the river-dolphin families Platanistidae, Iniidae, Pontoporiidae and probably the extinct Lipotidae (baiji or Chinese river dolphin).",
  "The tiger (Panthera tigris) is a large cat and a member of the genus Panthera native to Asia. It has a powerful, muscular body with a large head and paws, a long tail and orange fur with black, mostly vertical stripes.",
  "Penguins are a group of flightless, semi-aquatic birds which live almost exclusively in the Southern Hemisphere. They have highly adapted for life in the ocean: flippers instead of wings, streamlined bodies for swimming, and counter-shaded dark and white plumage.",
  "A honey bee (also spelled honeybee) is a eusocial flying insect from the genus Apis of the largest bee family, Apidae. All honey bees are nectarivorous pollinators native to mainland Afro-Eurasia, but human migrations since the Age of Discovery introduced multiple subspecies of the western honey bee into the New World and Australia, resulting in the current cosmopolitan distribution of honey bees in all continents except Antarctica.",
  "The cheetah (Acinonyx jubatus) is a lightly built, spotted cat characterised by a small rounded head, a short snout, black tear-like facial streaks, a deep chest, long thin legs and a long tail. Its slender, canine-like form is highly adapted for speed, contrasting sharply with other large felids.",
  "Owls are birds from the order Strigiformes, which includes over 200 species of mostly solitary and nocturnal birds of prey typified by an upright stance, a large broad head, binocular vision, binaural hearing, sharp talons and feathers adapted for silent flight.",
  "The gray wolf or grey wolf (Canis lupus) is a large canine native to wilderness and remote areas of Eurasia and North America. Adult wolves measure 105-160 cm (41-63 in) in length and 80-85 cm (31-33 in) at shoulder height.",
]

QUERIES = [
    "What are the different species of elephants?",
    "What adaptations help owls hunt at night?",
    "Where do tigers live and what do they look like?",
]

def main():
    engine = SearchEngine(model_name=MODEL_NAME)
    engine.add_documents(DOCUMENTS)

    for query in QUERIES:
        print(f"Query: {query}")
        results = engine.search(query, top_k=3)
        for i, res in enumerate(results, 1):
            print(f"{i}. [{res['score']:.3f}] {res['text'][:60]}...")
        print()

if __name__ == "__main__":
    main()
