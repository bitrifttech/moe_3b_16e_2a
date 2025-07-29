"""
Optimization-related training callbacks.
"""
import torch
import logging
from transformers import TrainerCallback

# Get logger
logger = logging.getLogger(__name__)

class LrKickOnceCallback(TrainerCallback):
    """Multiply learning rate by a factor once at a specific step to break plateaus."""
    
    def __init__(self, factor=3.0, at_step=0):
        """
        Args:
            factor: Multiplication factor for learning rate
            at_step: Step at which to apply the kick
        """
        self.factor = factor
        self.at_step = at_step
        self.done = False

    def on_step_end(self, args, state, control, **kwargs):
        """Apply learning rate kick at specified step."""
        if not self.done and state.global_step >= self.at_step:
            opt = kwargs.get("optimizer")
            sched = kwargs.get("lr_scheduler") 
            if opt and sched:
                # Boost current LR
                for g in opt.param_groups:
                    g["lr"] *= self.factor
                # Update scheduler base LRs so it doesn't reset
                if hasattr(sched, 'base_lrs'):
                    sched.base_lrs = [lr * self.factor for lr in sched.base_lrs]
                logger.info(f"🚀 LR kicked by ×{self.factor} at step {state.global_step}")
                self.done = True


class MoERoutingCallback(TrainerCallback):
    """Log MoE expert utilization statistics."""
    
    def __init__(self, log_every=500):
        """
        Args:
            log_every: Log routing stats every N steps
        """
        self.log_every = log_every
        
    def on_step_end(self, args, state, control, **kwargs):
        """Log MoE routing statistics at regular intervals."""
        if state.global_step % self.log_every == 0:
            logger.info(f"🔍 MoE Debug @ Step {state.global_step}: Callback triggered")
            model = kwargs.get("model")
            if model and hasattr(model, 'transformer'):
                self._log_routing_stats(model, state.global_step)
    
    def _log_routing_stats(self, model, step):
        """Extract and log expert routing statistics."""
        try:
            expert_counts = []
            total_tokens = 0
            aux_losses = []
            
            if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                # Check each MoE block
                for idx, block in enumerate(model.transformer.h):
                    if hasattr(block, 'mlp') and hasattr(block.mlp, 'moe'):
                        moe_layer = block.mlp.moe
                        
                        # Get routing statistics from Tutel MOELayer
                        if hasattr(moe_layer, 'dispatch_count') and moe_layer.dispatch_count is not None:
                            counts = moe_layer.dispatch_count.cpu().numpy()
                            expert_counts.append(counts)
                            total_tokens += counts.sum()
                        
                        # Get auxiliary loss (load balancing)
                        if hasattr(moe_layer, 'l_aux') and moe_layer.l_aux is not None:
                            aux_loss_val = moe_layer.l_aux if isinstance(moe_layer.l_aux, float) else moe_layer.l_aux.item()
                            aux_losses.append(aux_loss_val)
            
            if expert_counts:
                # Calculate utilization stats
                all_counts = torch.cat([torch.tensor(c, dtype=torch.float32) for c in expert_counts])
                total_routed = all_counts.sum().item()
                expert_util = (all_counts > 0).float().mean().item() * 100  # % of experts used
                load_balance = all_counts.std().item() / (all_counts.mean().item() + 1e-8)  # CV
                
                # Calculate per-layer stats
                layer_stats = []
                for i, counts in enumerate(expert_counts):
                    layer_util = (counts > 0).sum() / len(counts) * 100
                    layer_cv = counts.std() / (counts.mean() + 1e-8)
                    max_load = counts.max()
                    layer_stats.append(f"L{i*2}:{layer_util:.0f}%/{layer_cv:.2f}/{max_load}")
                
                avg_aux_loss = sum(aux_losses) / len(aux_losses) if aux_losses else 0
                
                logger.info(f"🧠 MoE Routing @ Step {step}: "
                          f"Overall Util: {expert_util:.1f}%, CV: {load_balance:.3f}, "
                          f"Aux Loss: {avg_aux_loss:.3f}, Total: {total_routed}")
                logger.info(f"📊 Per-Layer [Util%/CV/MaxLoad]: {' | '.join(layer_stats)}")
            else:
                logger.info(f"🔍 MoE Debug: No expert counts found - expert_counts is empty")
        except Exception as e:
            logger.error(f"MoE routing stats failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
