import re
import subprocess


class GPUSelector:
    """ Code from https://stackoverflow.com/questions/41634674/tensorflow-on-shared-gpus-how-to-automatically-select-the-one-that-is-unused """
    # Nvidia-smi GPU memory parsing.
    # Tested on nvidia-smi 370.23

    def run_command(self, cmd):
        """Run command, return output as string."""
        output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
        return output.decode("ascii")

    def list_available_gpus(self):
        """Returns list of available GPU ids."""
        output = self.run_command("nvidia-smi -L")
        # lines of the form GPU 0: TITAN X
        gpu_regex = re.compile(r"GPU (?P<gpu_id>\d+):")
        result = []
        for line in output.strip().split("\n"):
            m = gpu_regex.match(line)
            if m is None:
                return None
            assert m, "Couldnt parse " + line
            result.append(int(m.group("gpu_id")))
        return result

    def gpu_memory_map(self):
        """Returns map of GPU id to memory allocated on that GPU."""

        output = self.run_command("nvidia-smi")
        gpu_output = output[output.find("GPU Memory"):]
        # lines of the form
        # |    0      8734    C   python                                       11705MiB |
        memory_regex = re.compile(r"[|]\s+?(?P<gpu_id>\d+)\D+?(?P<pid>\d+).+[ ](?P<gpu_memory>\d+)MiB")
        rows = gpu_output.split("\n")
        if self.list_available_gpus() is None:
            return None
        result = {gpu_id: 0 for gpu_id in self.list_available_gpus()}
        for row in gpu_output.split("\n"):
            m = memory_regex.search(row)
            if not m:
                continue
            gpu_id = int(m.group("gpu_id"))
            gpu_memory = int(m.group("gpu_memory"))
            result[gpu_id] += gpu_memory
        return result

    def pick_gpu_lowest_memory(self):
        """Returns GPU with the least allocated memory"""
        if self.gpu_memory_map() is None:
            print("Couldn't find lowest memory GPU")
            return None

        best_gpu = ""
        memory_gpu_map = [(memory, gpu_id) for (gpu_id, memory) in self.gpu_memory_map().items()]
        best_memory, best_gpu = sorted(memory_gpu_map)[0]
        print(f"Using CUDA_VISIBLE_DEVICES {best_gpu}")

        return best_gpu