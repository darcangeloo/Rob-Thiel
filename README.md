![logo](assets/readme.jpg)


# RobThiel

Esperimento didattico di fine-tuning QLoRA su Qwen2.5-7B-Instruct, dataset startup/marketing ispirato a *Zero to One* di Peter Thiel (da cui il nome).

## Obiettivo

Questo progetto non nasce per produrre un modello da usare per consigli di business reali. L'obiettivo è imparare la pipeline di fine-tuning end-to-end: preparazione dei dati in formato istruzione-risposta, quantizzazione 4-bit, configurazione LoRA, training con monitoraggio della loss di validazione, e valutazione tramite confronto diretto tra modello base e modello fine-tuned.

Il criterio di successo non è "il modello dà consigli affidabili", ma "la pipeline funziona correttamente e produce un cambiamento misurabile e comprensibile nel comportamento del modello".

## Dataset

- **Dominio**: startup, zero-to-one, marketing, positioning
- **Dimensione**: ~460 coppie istruzione-risposta (formato Alpaca: `instruction`, `input`, `output`, `category`)
- **Generazione**: contenuto generato con assistenza LLM a partire da temi definiti manualmente, poi validato a campione per correttezza fattuale e coerenza di stile
- **Limiti**: dataset di dimensioni ridotte, copertura del dominio parziale. Il modello fine-tuned tende a riflettere lo stile e i pattern del dataset più che a dimostrare capacità di ragionamento originale su casi non visti in training.

## Stack tecnico

- **Modello base**: Qwen2.5-7B-Instruct
- **Metodo**: QLoRA (LoRA su modello quantizzato in 4-bit tramite `bitsandbytes`)
- **Librerie**: `transformers`, `peft`, `trl` (`SFTTrainer`), `bitsandbytes`, `accelerate`, `datasets`
- **Hardware**: NVIDIA RTX 5060 Laptop, 8GB VRAM
- **Configurazione LoRA**: `r=16`, `lora_alpha=16`, `target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]`, `lora_dropout=0.1`

## Setup e riproduzione

```bash
git clone <repo-url>
cd RobThiel
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

Serve un token HuggingFace con accesso al modello (per Qwen2.5-7B-Instruct non è gated, non serve approvazione).

Per lanciare il training:

```bash
python train.py
```

Per testare il modello fine-tuned:

```bash
python inference.py
```

## Risultati

Training di 3 epoche su ~390 esempi (train split), ~70 esempi (validation split), batch size effettivo 12 (1 × 12 gradient accumulation).

| Step | eval_loss | eval_mean_token_accuracy |
|------|-----------|---------------------------|
| 10   | 2.21      | 0.565                     |
| 123 (finale) | 1.567 | 0.634                  |

La eval loss scende in modo monotono per tutta la run, senza segni di overfitting entro le 3 epoche (la curva risultava ancora in discesa a fine training).

*(Inserire qui screenshot del confronto tra output del modello base e del modello fine-tuned sulla stessa domanda)*

## Limiti noti

- Dataset di dimensioni ridotte (poche centinaia di esempi): il modello generalizza poco oltre i pattern visti in training.
- Le risposte riflettono principalmente lo stile del dataset, non un ragionamento originale verificato.
- Nessuna validazione umana estesa delle risposte generate in inferenza oltre a test manuali su singole domande.
- Non è uno strumento pensato per l'uso in produzione o per decisioni di business reali.

## Licenza

*(da specificare)*
