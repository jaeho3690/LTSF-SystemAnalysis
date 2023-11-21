    # def update_system_related_info(self):
    #     """Update system related info to self.output_dict"""
    #     # get device info
    #     device_properties = torch.cuda.get_device_properties(0)
    #     device_name = device_properties.name
    #     device_total_memory = device_properties.total_memory / 1e9
    #     num_threads = torch.get_num_threads()

    #     # get system info
    #     python_ver = platform.python_version()
    #     pytorch_ver = torch.__version__
    #     pytorch_lightning_ver = L.__version__

    #     # get model info
    #     model_size_mb = L.utilities.memory.get_model_size_mb(self.model)

    #     # get system info
    #     max_memory_mb = torch.cuda.max_memory_allocated() / 1e6
    #     print(
    #         f"device_name: {device_name}, device_total_memory: {device_total_memory}, num_threads: {num_threads}, max_memory_mb: {max_memory_mb}"
    #     )
    #     # update as key to self.output_dict
    #     self.output_dict.update(
    #         {
    #             "device_name": device_name,
    #             "device_total_memory": device_total_memory,
    #             "num_threads": num_threads,
    #             "model_size_mb": model_size_mb,
    #             "max_memory_mb": max_memory_mb,
    #             "python_ver": python_ver,
    #             "pytorch_ver": pytorch_ver,
    #             "pytorch_lightning_ver": pytorch_lightning_ver,
    #         }
    #     )