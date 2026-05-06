#!/bin/bash

# ==============================================================================
# SCRIPT DE ORQUESTACIÓN: QFL-ADAPTIVE (SBRC 2026)
# ==============================================================================

# 1. Configuración de Entorno
# source ../env/bin/activate  # Descomentar si usas Virtualenv
echo ">> Activando entorno virtual..."
source ../ven/bin/activate

# Verificación de seguridad (Opcional: avisa si falla)
if [ $? -ne 0 ]; then
    echo "ERROR: No se pudo activar el entorno en ../env/bin/activate"
    echo "Verifica que la carpeta 'env' exista en el directorio superior."
    exit 1
fi

OUTPUT_DIR="./resultados_sbrc_2026"
mkdir -p $OUTPUT_DIR

# 2. Configuración Visual
IDIOMA="pt"          # Opciones: "en" (Inglés), "pt" (Portugués)
FONT_SIZE=14         # Tamaño de letra grande para legibilidad en papers

# 3. Hiperparámetros del Experimento
ROUNDS=25            # Rondas de comunicación por experimento
CLIENTS=10            # Número de clientes

# DATASETS: Lista de datos a probar (Clásicos y Médicos)
DATASETS=("mnist" "fashion" "pneumonia" "breast")

# UMBRALES SLIMMABLE (--channel_threshold): 
# 0.3 = Poda moderada (Solo poda si la red es muy mala)
# 0.8 = Poda agresiva (Poda a menos que la red sea perfecta)
#THRESHOLDS=(0.4 0.8)
# Estrategia: Comparar "Conservador" vs "Agresivo".
# 0.3: Solo comprime si la red es MUY mala (Prioriza Precisión).
# 0.7: Comprime ante cualquier duda (Prioriza Ahorro de Banda).
#THRESHOLDS=(0.3 0.7)# Estrategia: Comparar "Conservador" vs "Agresivo".
# 0.3: Solo comprime si la red es MUY mala (Prioriza Precisión).
# 0.7: Comprime ante cualquier duda (Prioriza Ahorro de Banda).
THRESHOLDS=(0.3 0.7)




# 0.0: Ideal
# 0.2: 4G/WiFi
# 0.5: IoT/Congestionado
# 0.8: Hostil/Intermitente
BAD_PROBS=(0.0 0.2 0.5 0.8)


# PROBABILIDAD DE FALLO (--bad_channel_prob):
# 0.2 = Red mayormente estable (20% fallo)
# 0.6 = Red hostil (60% fallo) - Prueba de Resiliencia
#BAD_PROBS=(0.2 0.6)

echo "##########################################################"
echo "  INICIANDO BATCH DE EXPERIMENTOS QFL-ADAPTIVE"
echo "  Idioma: $IDIOMA | Salida: $OUTPUT_DIR"
echo "##########################################################"

for dataset in "${DATASETS[@]}"; do
    for thresh in "${THRESHOLDS[@]}"; do
        for prob in "${BAD_PROBS[@]}"; do
            
            echo ""
            echo ">> [EJECUTANDO]: Dataset=$dataset | Thresh=$thresh | BadProb=$prob"
            
            python main_qfl.py \
                --dataset $dataset \
                --channel_threshold $thresh \
                --bad_channel_prob $prob \
                --rounds $ROUNDS \
                --clients $CLIENTS \
                --output_dir $OUTPUT_DIR \
                --language $IDIOMA \
                --font_size $FONT_SIZE
                
            echo ">> [OK] Completado."
            
        done
    done
done

echo ""
echo "##########################################################"
echo "  TODO FINALIZADO. RESULTADOS GENERADOS:"
echo "  1. CSVs de Métricas: $OUTPUT_DIR/*.csv"
echo "  2. Gráficos ($IDIOMA): $OUTPUT_DIR/images/*.png"
echo "##########################################################"
