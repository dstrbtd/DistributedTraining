import bittensor as bt


def log_peerid_to_chain(self):
    try:
        metadata = {
            "peer_id": self.dht.peer_id.to_base58(),
            "model_huggingface_id": self.config.neuron.local_model_name,
        }
        self.subtensor.commit(self.wallet, self.config.netuid, str(metadata))
        self.peer_id_logged_to_chain = True
        self.logger.info(f"Metadata dict {metadata} succesfully logged to chain.")
    except Exception:
        self.peer_id_logged_to_chain = False
        self.logger.debug(
            "Unable to log DHT PeerID to chain. Retrying on the next step."
        )
