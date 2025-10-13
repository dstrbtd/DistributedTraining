from distributed_training import __run__


def log_peerid_to_chain(self):
    if self.master:
        try:
            metadata = {
                "peer_id": self.dht.peer_id.to_base58(),
                "model_huggingface_id": self.config.neuron.local_model_name,
                "r2_account_id": self.config.r2.account_id,
                "r2_bucket_name": self.config.r2.bucket_name,
                "r2_secret_access_key_id": self.config.r2.read.access_key_id,
                "r2_secret_access_key": self.config.r2.read.secret_access_key,
                "run": __run__,
                "outer_step": str(self.local_progress.epoch),
                "inner_step": str(self.local_progress.inner_step),
            }
            metadata = (
                self.config.r2.account_id
                + self.config.r2.read.access_key_id
                + self.config.r2.read.secret_access_key
            )
            self.subtensor.commit(self.wallet, self.config.netuid, str(metadata))
            self.peer_id_logged_to_chain = True
            self.logger.info(f"Metadata dict succesfully logged to chain.")
        except Exception as e:
            self.peer_id_logged_to_chain = False
            self.logger.info(
                f"Unable to log DHT PeerID to chain due to error {e}. Retrying on the next step."
            )
