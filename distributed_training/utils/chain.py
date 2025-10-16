from distributed_training import __run__


def log_peerid_to_chain(self):
    if self.master:
        try:
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
                f"Unable to log bucket data to chain due to error {e}. Retrying on the next step."
            )
