from distributed_training import __run__


def log_r2_to_chain(self):
    if self.master:
        try:
            metadata = (
                self.config.r2.account_id
                + self.config.r2.read.access_key_id
                + self.config.r2.read.secret_access_key
            )
            self.subtensor.commit(self.wallet, self.config.netuid, str(metadata))
            self.r2_credentials_logged_to_chain = True
            self.logger.info(f"Metadata Dict Succesfully Logged To Chain.")
        except Exception as e:
            self.peer_id_logged_to_chain = False
            self.logger.info(
                f"Unable To Log Bucket Data To Chain Due To Error {e}. Retrying On The Next Step."
            )
