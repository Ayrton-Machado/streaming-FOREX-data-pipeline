# Jupyter Notebook Configuration
c = get_config()

# Interface básica
c.NotebookApp.token = 'forex123'
c.NotebookApp.password = ''
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.port = 8888
c.NotebookApp.allow_root = True
c.NotebookApp.open_browser = False

# Segurança
c.NotebookApp.allow_origin = '*'
c.NotebookApp.tornado_settings = {
    'headers': {
        'Content-Security-Policy': "frame-ancestors 'self' *"
    }
}

# Diretórios
c.NotebookApp.notebook_dir = '/home/jovyan/work'

# Kernels disponíveis
c.KernelSpecManager.ensure_native_kernel = True