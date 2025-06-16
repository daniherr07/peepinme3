import Chat from "./components/Chat"
import styles from "./page.module.css"

const Logo = () => (
  <svg
    width="48"
    height="48"
    viewBox="0 0 48 48"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    className={styles.logo}
  >
    <path
      d="M24 4C15.1634 4 8 11.1634 8 20C8 26.6865 12.7938 32.5028 19 34.414V42C19 42.5523 19.4477 43 20 43H28C28.5523 43 29 42.5523 29 42V34.414C35.2062 32.5028 40 26.6865 40 20C40 11.1634 32.8366 4 24 4Z"
      fill="#4A90E2"
    />
    <circle cx="24" cy="20" r="6" fill="white" />
  </svg>
)

export default function Home() {
  return (
    <main className={styles.main}>
      <div className={styles.container}>
        <header className={styles.header}>
          <div className={styles.logoContainer}>
            <Logo />
          </div>
          <h1 className={styles.title}>PeepInMe</h1>
          <p className={styles.subtitle}>Find stores in Costa Rica</p>
        </header>

        <Chat />

        <footer className={styles.footer}>
          <p>© {new Date().getFullYear()} PeepInMe. Pura Vida!</p>
        </footer>
      </div>
    </main>
  )
}
