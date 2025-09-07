"""
Abstract Base Pipeline for Federated Learning
Provides unified interface for FHE and Plain Text implementations
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
from datetime import datetime

@dataclass
class PipelineConfig:
    """Unified configuration for all pipeline types"""
    rounds: int = 10
    clients: int = 10
    min_samples_per_client: int = 50
    test_size: float = 0.2
    random_state: int = 42
    data_path: str = "data/health_fitness_data.csv"
    
    # Encryption-specific settings
    encryption_enabled: bool = False
    encryption_scheme: str = "CKKS"
    polynomial_degree: int = 8192
    scale_bits: int = 40
    
    # Performance tracking
    track_detailed_metrics: bool = True
    save_intermediate_results: bool = True
    enable_profiling: bool = False

@dataclass
class RoundResult:
    """Standardized result structure for each round"""
    round_id: int
    timestamp: datetime
    
    # Model performance
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    
    # Timing metrics
    training_time: float
    aggregation_time: float
    encryption_time: Optional[float] = None
    decryption_time: Optional[float] = None
    communication_time: Optional[float] = None
    
    # Resource metrics
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    energy_consumption: Optional[float] = None
    
    # Communication metrics
    bytes_transferred: Optional[float] = None
    communication_overhead: Optional[float] = None
    
    # Model state
    model_weights: Optional[np.ndarray] = None
    model_bias: Optional[float] = None
    weight_norm: Optional[float] = None
    
    # Metadata
    is_encrypted: bool = False
    encryption_scheme: Optional[str] = None
    convergence_indicator: Optional[bool] = None

@dataclass
class ExperimentResult:
    """Complete experiment results"""
    experiment_id: str
    pipeline_type: str  # "FHE_CKKS" or "PLAINTEXT"
    config: PipelineConfig
    start_time: datetime
    end_time: datetime
    total_duration: float
    
    # Results
    round_results: List[RoundResult]
    final_model_weights: np.ndarray
    final_model_bias: float
    
    # Summary metrics
    final_accuracy: float
    best_accuracy: float
    accuracy_improvement: float
    convergence_round: Optional[int]
    
    # Performance summary
    avg_training_time: float
    avg_aggregation_time: float
    total_communication_bytes: float
    total_energy_consumption: Optional[float]
    
    # Statistical significance
    confidence_interval: Optional[Tuple[float, float]] = None
    p_value: Optional[float] = None

class BaseFederatedLearningPipeline(ABC):
    """
    Abstract base class for federated learning pipelines
    Ensures consistent interface for fair comparison
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.clients_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.global_model = None
        self.round_results: List[RoundResult] = []
        self.experiment_id = f"{self.get_pipeline_type()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    @abstractmethod
    def get_pipeline_type(self) -> str:
        """Return pipeline type identifier"""
        pass
    
    @abstractmethod
    def initialize_global_model(self) -> Any:
        """Initialize the global model"""
        pass
    
    @abstractmethod
    def train_local_model(self, client_id: str, X: np.ndarray, y: np.ndarray) -> Any:
        """Train local model for a client"""
        pass
    
    @abstractmethod
    def aggregate_updates(self, local_updates: List[Any], sample_counts: List[int]) -> Any:
        """Aggregate local updates into global model"""
        pass
    
    @abstractmethod
    def evaluate_model(self, model: Any, test_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """Evaluate model performance"""
        pass
    
    @abstractmethod
    def update_global_model(self, aggregated_update: Any) -> None:
        """Update global model with aggregated update"""
        pass
    
    def load_and_prepare_data(self) -> None:
        """Load and prepare data for federated learning"""
        from src.fl import DataProcessor
        
        processor = DataProcessor(self.config)
        df, feature_columns = processor.load_health_fitness_data(self.config.data_path)
        
        if df is None:
            raise ValueError(f"Failed to load data from {self.config.data_path}")
        
        self.clients_data = processor.create_client_datasets(df)
        self.clients_data = processor.scale_client_data(self.clients_data)
        
        print(f"✅ Data loaded: {len(self.clients_data)} clients, {len(feature_columns)} features")
    
    def run_federated_learning(self) -> ExperimentResult:
        """Run complete federated learning experiment"""
        print(f"\n🚀 Starting {self.get_pipeline_type()} Federated Learning")
        print("=" * 70)
        
        start_time = datetime.now()
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Initialize global model
        self.global_model = self.initialize_global_model()
        
        # Run federated learning rounds
        for round_id in range(self.config.rounds):
            print(f"\n📊 Round {round_id + 1}/{self.config.rounds}")
            
            round_result = self._execute_round(round_id)
            self.round_results.append(round_result)
            
            # Print round summary
            self._print_round_summary(round_result)
        
        # Create final experiment result
        end_time = datetime.now()
        experiment_result = self._create_experiment_result(start_time, end_time)
        
        # Save results
        self._save_experiment_results(experiment_result)
        
        return experiment_result
    
    def _execute_round(self, round_id: int) -> RoundResult:
        """Execute a single federated learning round"""
        round_start = datetime.now()
        
        # Collect local updates
        local_updates = []
        sample_counts = []
        training_times = []
        
        for client_id, (X, y) in self.clients_data.items():
            train_start = datetime.now()
            local_model = self.train_local_model(client_id, X, y)
            training_time = (datetime.now() - train_start).total_seconds()
            
            local_updates.append(local_model)
            sample_counts.append(len(X))
            training_times.append(training_time)
        
        # Aggregate updates
        agg_start = datetime.now()
        aggregated_update = self.aggregate_updates(local_updates, sample_counts)
        aggregation_time = (datetime.now() - agg_start).total_seconds()
        
        # Update global model
        self.update_global_model(aggregated_update)
        
        # Evaluate model
        test_data = self._create_test_data()
        metrics = self.evaluate_model(self.global_model, test_data)
        
        # Create round result
        round_result = RoundResult(
            round_id=round_id,
            timestamp=round_start,
            accuracy=metrics['accuracy'],
            f1_score=metrics['f1_score'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            training_time=np.mean(training_times),
            aggregation_time=aggregation_time,
            is_encrypted=self.config.encryption_enabled,
            encryption_scheme=self.config.encryption_scheme if self.config.encryption_enabled else None
        )
        
        return round_result
    
    def _create_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create test data for evaluation"""
        # Combine data from all clients for testing
        all_x_data = []
        all_y_data = []
        
        for X, y in self.clients_data.values():
            test_size = min(50, len(X) // 2)  # Use subset for testing
            all_x_data.append(X[:test_size])
            all_y_data.append(y[:test_size])
        
        return np.vstack(all_x_data), np.hstack(all_y_data)
    
    def _print_round_summary(self, result: RoundResult) -> None:
        """Print summary for a round"""
        print(f"  📈 Accuracy: {result.accuracy:.4f} ({result.accuracy*100:.2f}%)")
        print(f"  🎯 F1 Score: {result.f1_score:.4f} ({result.f1_score*100:.2f}%)")
        print(f"  ⏱️  Training Time: {result.training_time:.4f}s")
        print(f"  🔄 Aggregation Time: {result.aggregation_time:.4f}s")
        
        if result.is_encrypted:
            print(f"  🔐 Encryption: {result.encryption_scheme}")
            if result.encryption_time:
                print(f"  🔒 Encryption Time: {result.encryption_time:.4f}s")
    
    def _create_experiment_result(self, start_time: datetime, end_time: datetime) -> ExperimentResult:
        """Create final experiment result"""
        total_duration = (end_time - start_time).total_seconds()
        
        # Calculate summary metrics
        final_accuracy = self.round_results[-1].accuracy
        best_accuracy = max(r.accuracy for r in self.round_results)
        accuracy_improvement = final_accuracy - self.round_results[0].accuracy
        
        avg_training_time = np.mean([r.training_time for r in self.round_results])
        avg_aggregation_time = np.mean([r.aggregation_time for r in self.round_results])
        
        # Get final model state
        final_weights, final_bias = self._extract_model_state()
        
        return ExperimentResult(
            experiment_id=self.experiment_id,
            pipeline_type=self.get_pipeline_type(),
            config=self.config,
            start_time=start_time,
            end_time=end_time,
            total_duration=total_duration,
            round_results=self.round_results,
            final_model_weights=final_weights,
            final_model_bias=final_bias,
            final_accuracy=final_accuracy,
            best_accuracy=best_accuracy,
            accuracy_improvement=accuracy_improvement,
            convergence_round=None,
            avg_training_time=avg_training_time,
            avg_aggregation_time=avg_aggregation_time,
            total_communication_bytes=0.0,
            total_energy_consumption=None
        )
    
    @abstractmethod
    def _extract_model_state(self) -> Tuple[np.ndarray, float]:
        """Extract final model weights and bias"""
        pass
    
    def _save_experiment_results(self, result: ExperimentResult) -> None:
        """Save experiment results to file"""
        import json
        import os
        
        os.makedirs("experiments", exist_ok=True)
        
        # Convert to serializable format
        result_dict = {
            'experiment_id': result.experiment_id,
            'pipeline_type': result.pipeline_type,
            'config': {
                'rounds': result.config.rounds,
                'clients': result.config.clients,
                'encryption_enabled': result.config.encryption_enabled,
                'encryption_scheme': result.config.encryption_scheme
            },
            'start_time': result.start_time.isoformat(),
            'end_time': result.end_time.isoformat(),
            'total_duration': result.total_duration,
            'final_accuracy': result.final_accuracy,
            'best_accuracy': result.best_accuracy,
            'accuracy_improvement': result.accuracy_improvement,
            'avg_training_time': result.avg_training_time,
            'avg_aggregation_time': result.avg_aggregation_time,
            'round_results': [
                {
                    'round_id': r.round_id,
                    'accuracy': r.accuracy,
                    'f1_score': r.f1_score,
                    'precision': r.precision,
                    'recall': r.recall,
                    'training_time': r.training_time,
                    'aggregation_time': r.aggregation_time,
                    'is_encrypted': r.is_encrypted
                }
                for r in result.round_results
            ]
        }
        
        filename = f"experiments/{result.experiment_id}.json"
        with open(filename, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        print(f"💾 Results saved to: {filename}")
