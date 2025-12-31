import toast from 'react-hot-toast'

export const showSuccess = (message: string) => {
  toast.success(message)
}

export const showError = (message: string) => {
  toast.error(message)
}

export const showInfo = (message: string) => {
  toast(message, {
    icon: 'ℹ️',
  })
}

export const showWarning = (message: string) => {
  toast(message, {
    icon: '⚠️',
    style: {
      background: '#1e293b',
      color: '#f1f5f9',
      border: '1px solid #f59e0b',
    },
  })
}

export const showLoading = (message: string) => {
  return toast.loading(message)
}

export const dismissToast = (toastId: string) => {
  toast.dismiss(toastId)
}
