/**
 * AnimatedList Component
 *
 * Provides staggered entrance animations for list items.
 * Uses Framer Motion for smooth, performant animations.
 */

import React, { Children, ReactNode } from 'react';
import { motion, Variants, AnimatePresence } from 'framer-motion';

interface AnimatedListProps {
  children: ReactNode;
  staggerDelay?: number;
  initialDelay?: number;
  className?: string;
  itemClassName?: string;
  animation?: 'fade' | 'slide-up' | 'slide-right' | 'scale';
  layout?: boolean;
}

const animations: Record<string, Variants> = {
  'fade': {
    hidden: { opacity: 0 },
    visible: { opacity: 1 },
    exit: { opacity: 0 },
  },
  'slide-up': {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: -10 },
  },
  'slide-right': {
    hidden: { opacity: 0, x: -20 },
    visible: { opacity: 1, x: 0 },
    exit: { opacity: 0, x: 20 },
  },
  'scale': {
    hidden: { opacity: 0, scale: 0.9 },
    visible: { opacity: 1, scale: 1 },
    exit: { opacity: 0, scale: 0.9 },
  },
};

export const AnimatedList: React.FC<AnimatedListProps> = ({
  children,
  staggerDelay = 0.05,
  initialDelay = 0,
  className = '',
  itemClassName = '',
  animation = 'slide-up',
  layout = false,
}) => {
  const childArray = Children.toArray(children);
  const itemVariants = animations[animation];

  const customContainerVariants: Variants = {
    hidden: { opacity: 1 },
    visible: {
      opacity: 1,
      transition: {
        delayChildren: initialDelay,
        staggerChildren: staggerDelay,
      },
    },
  };

  return (
    <motion.div
      className={className}
      variants={customContainerVariants}
      initial="hidden"
      animate="visible"
    >
      <AnimatePresence mode="popLayout">
        {childArray.map((child, index) => (
          <motion.div
            key={index}
            className={itemClassName}
            variants={itemVariants}
            layout={layout}
            transition={{
              duration: 0.3,
              ease: [0.16, 1, 0.3, 1],
            }}
          >
            {child}
          </motion.div>
        ))}
      </AnimatePresence>
    </motion.div>
  );
};

/**
 * AnimatedListItem Component
 *
 * Individual animated item for use within lists.
 * Can be used standalone without AnimatedList wrapper.
 */
interface AnimatedListItemProps {
  children: ReactNode;
  delay?: number;
  animation?: 'fade' | 'slide-up' | 'slide-right' | 'scale';
  className?: string;
  layout?: boolean;
}

export const AnimatedListItem: React.FC<AnimatedListItemProps> = ({
  children,
  delay = 0,
  animation = 'slide-up',
  className = '',
  layout = false,
}) => {
  const itemVariants = animations[animation];

  return (
    <motion.div
      className={className}
      initial="hidden"
      animate="visible"
      exit="exit"
      variants={itemVariants}
      layout={layout}
      transition={{
        duration: 0.3,
        delay,
        ease: [0.16, 1, 0.3, 1],
      }}
    >
      {children}
    </motion.div>
  );
};

export default AnimatedList;
