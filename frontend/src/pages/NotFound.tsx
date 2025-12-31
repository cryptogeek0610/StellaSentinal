import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'

export default function NotFound() {
  return (
    <div className="flex items-center justify-center min-h-[calc(100vh-200px)]">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center"
      >
        <div className="mb-8">
          <div className="text-8xl font-bold text-cyan-400 mb-4">404</div>
          <h1 className="text-3xl font-bold text-white mb-2">Page Not Found</h1>
          <p className="text-gray-400 text-lg">
            The page you're looking for doesn't exist or has been moved.
          </p>
        </div>

        <div className="space-x-4">
          <Link
            to="/"
            className="inline-block px-6 py-3 bg-cyan-600 hover:bg-cyan-700 text-white rounded-lg transition-colors"
          >
            Go to Dashboard
          </Link>
          <Link
            to="/investigations"
            className="inline-block px-6 py-3 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors"
          >
            View Investigations
          </Link>
        </div>
      </motion.div>
    </div>
  )
}
