from serenity_trainer.util.config.ConceptConfig import read_concepts

def main() raises:
    var cs = read_concepts(String("/home/alex/Serenity/training_concepts/alina_baseline_concepts.json"))
    print("n concepts =", len(cs), " (OT 1)")
    ref c0 = cs[0]
    print("name =", c0.name, " (OT alina_baseline)")
    print("path =", c0.path, " (OT /home/alex/datasets/AlinaAignatova)")
    print("enabled =", c0.enabled, " (OT True)")
    print("balancing =", c0.balancing, "  text.prompt_source =", c0.text.prompt_source)
    var ok = (len(cs) == 1) and (c0.name == String("alina_baseline")) and (c0.path == String("/home/alex/datasets/AlinaAignatova")) and c0.enabled
    print("CONCEPT READER PARITY", "PASS" if ok else "FAIL")
