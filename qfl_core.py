import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import copy

# ==============================================================================
# 1. MODELO: SLIMMABLE VARIATIONAL QUANTUM CIRCUIT (VQC)
# 
# Referencia Interna: Sección 3.1 "Modelo Local: VQC" 
# Referencia Externa: Park et al. (2025). "Quantum federated learning with 
#                     pole-angle quantum local training" 
# ==============================================================================
class SlimmableVQC(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=64, output_dim=2):
        super(SlimmableVQC, self).__init__()
        
        # PARTE 'PHI' (ANGLES): 
        # Representa las rotaciones del circuito (entrenamiento costoso).
        # En simulación clásica, actúa como extractor de características.
        self.phi_body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # PARTE 'THETA' (POLES):
        # Representa los parámetros de medición (entrenamiento ligero).
        # Esta es la única parte que se transmite en condiciones de red adversas[cite: 54].
        self.theta_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten
        features = self.phi_body(x)
        return self.theta_head(features)

    def get_partial_parameters(self):
        """Extrae solo la cabeza (Polos) para transmisión eficiente[cite: 32]."""
        return self.theta_head.state_dict()

# ==============================================================================
# 2. CLIENTE FEDERADO (PERSONALIZADO Y RESILIENTE)
# ==============================================================================
class QFLClient:
    def __init__(self, client_id, dataset, device='cpu', batch_size=32, lr=0.005):
        self.client_id = client_id
        self.device = device
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model = SlimmableVQC().to(device)
        self.local_theta_backup = None
        self.lr = lr

    # --------------------------------------------------------------------------
    # Lógica de Personalización (wp-QFL)
    # Referencia Interna: Sección 3.3 
    # Referencia Externa: Gurung & Pokhrel (2025). "Weighted Personalized QFL" 
    # --------------------------------------------------------------------------
    def update_and_personalize(self, global_weights, alpha_wp=0.8):
        """
        Calcula Theta_k = alpha * Global + (1-alpha) * Local_Anterior.
        Mitiga el efecto Non-IID preservando características locales[cite: 58].
        """
        if self.local_theta_backup is not None:
            new_state_dict = self.model.state_dict()
            for k in global_weights.keys():
                if k in new_state_dict:
                    # Interpolación lineal de pesos (Eq. 1 del Paper)
                    new_state_dict[k] = alpha_wp * global_weights[k] + (1 - alpha_wp) * new_state_dict[k]
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(global_weights)

    def train(self):
        """Entrenamiento local estándar (Simulación de VQC optimizado)"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        epoch_loss = 0
        
        # 1 época local por ronda para simular dinámica rápida de FL
        for _ in range(1): 
            for data, target in self.loader:
                data, target = data.to(self.device).float(), target.to(self.device).long()
                if target.dim() > 1: target = target.squeeze()
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
        # Backup para la personalización de la siguiente ronda
        self.local_theta_backup = copy.deepcopy(self.model.state_dict())
        return epoch_loss / len(self.loader)

    # --------------------------------------------------------------------------
    # Lógica de Comunicación Adaptativa (Slimmable) + Seguridad
    # Referencia Interna: Sección 3.2  y 3.4 
    # Referencia Externa: Park et al. (2025)  y Liu et al. (2025) 
    # --------------------------------------------------------------------------
    def adaptive_upload(self, bad_channel_prob, channel_threshold):
        """
        Decide qué enviar basado en la calidad del canal simulado.
        Args:
            bad_channel_prob: Probabilidad de estar en un entorno ruidoso/hostil.
            channel_threshold: Umbral de calidad bajo el cual se activa el modo Slim.
        """
        # 1. Simulación Estocástica del Canal [cite: 68]
        if np.random.rand() < bad_channel_prob:
            current_quality = 0.2  # Canal "Malo" (Ruido/Congestión)
        else:
            current_quality = 0.9  # Canal "Bueno" (Estable)

        full_size = sum(p.numel() for p in self.model.state_dict().values()) * 4

        # 2. Decisión Slimmable [cite: 54]
        if current_quality < channel_threshold:
            # Estrategia: Solo enviar Polos (Theta) -> Ahorro masivo
            params = self.model.get_partial_parameters()
            p_type = 'pole'
            sent_size = sum(p.numel() for p in params.values()) * 4
        else:
            # Estrategia: Enviar Todo (Full)
            params = self.model.state_dict()
            p_type = 'full'
            sent_size = full_size

        # (Nota: La simulación de QKD se asume implícita en la seguridad del canal aquí)
        return params, p_type, sent_size, full_size

# ==============================================================================
# 3. SERVIDOR (AGREGACIÓN HÍBRIDA)
# Referencia Interna: Sección 3.3 
# ==============================================================================
def hybrid_aggregation(global_model, updates):
    """
    Agrega actualizaciones mixtas (Partial/Pole y Full).
    Asegura que el modelo global converja incluso con updates incompletos.
    """
    global_dict = global_model.state_dict()
    keys_sum = {k: torch.zeros_like(v).float() for k, v in global_dict.items()}
    keys_count = {k: 0 for k in global_dict.keys()}

    for params, p_type in updates:
        for k, v in params.items():
            if k in global_dict:
                keys_sum[k] += v
                keys_count[k] += 1

    # Promedio ponderado donde haya datos disponibles
    for k in global_dict.keys():
        if keys_count[k] > 0:
            global_dict[k] = keys_sum[k] / keys_count[k]

    global_model.load_state_dict(global_dict)
    return global_model
