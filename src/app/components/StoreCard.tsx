import type { Store } from "../lib/chatbot"
import styles from "./StoreCard.module.css"

interface StoreCardProps {
  store: Store
}

const StoreCard = ({ store }: StoreCardProps) => {
  return (
    <div className={styles.card}>
      <div className={styles.header}>
        <h3 className={styles.storeName}>{store.name}</h3>
        <span className={styles.category}>{store.category}</span>
      </div>

      <div className={styles.details}>
        <div className={styles.detail}>
          <span className={styles.icon}>📍</span>
          <span>
            {store.location.city}, {store.location.province}
          </span>
        </div>
        <div className={styles.detail}>
          <span className={styles.icon}>🕒</span>
          <span>{store.hours}</span>
        </div>
        {store.contact !== "N/A" && (
          <div className={styles.detail}>
            <span className={styles.icon}>📞</span>
            <span>{store.contact}</span>
          </div>
        )}
      </div>

      <div className={styles.products}>
        <p className={styles.productsLabel}>Available products:</p>
        <div className={styles.productTags}>
          {store.product_types.slice(0, 5).map((product, index) => (
            <span key={index} className={styles.tag}>
              {product}
            </span>
          ))}
          {store.product_types.length > 5 && (
            <span className={styles.moreTag}>+{store.product_types.length - 5} more</span>
          )}
        </div>
      </div>
    </div>
  )
}

export default StoreCard
