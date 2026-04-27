import { useEffect, useState } from 'react'
import { NAV_ITEMS } from '../app/constants'
import type { PageId, SectionId } from '../app/types'

export function useWorkspaceNavigation() {
  const [activePageId, setActivePageId] = useState<PageId>('home')
  const [activeSectionId, setActiveSectionId] = useState<SectionId>('input-import')
  const [isSidebarExpanded, setIsSidebarExpanded] = useState(false)

  useEffect(() => {
    if (typeof window === 'undefined' || typeof IntersectionObserver === 'undefined') {
      return
    }

    if (activePageId !== 'workspace') {
      return
    }

    const sections = NAV_ITEMS
      .map((item) => document.getElementById(item.id))
      .filter((section): section is HTMLElement => section instanceof HTMLElement)

    if (sections.length === 0) {
      return
    }

    const visibleSections = new Map<SectionId, number>()
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          const sectionId = entry.target.id as SectionId
          if (entry.isIntersecting) {
            visibleSections.set(sectionId, entry.intersectionRatio)
            return
          }

          visibleSections.delete(sectionId)
        })

        const nextActiveSection = NAV_ITEMS
          .map((item, index) => ({
            id: item.id,
            index,
            ratio: visibleSections.get(item.id) ?? -1,
          }))
          .filter((item) => item.ratio >= 0)
          .sort((left, right) => right.ratio - left.ratio || left.index - right.index)[0]

        if (nextActiveSection) {
          setActiveSectionId((currentValue) =>
            currentValue === nextActiveSection.id ? currentValue : nextActiveSection.id)
        }
      },
      {
        rootMargin: '-18% 0px -54% 0px',
        threshold: [0.12, 0.3, 0.48, 0.72],
      },
    )

    sections.forEach((section) => {
      observer.observe(section)
    })

    return () => {
      observer.disconnect()
    }
  }, [activePageId])

  function handleNavigate(sectionId: SectionId) {
    const section = document.getElementById(sectionId)
    if (!section) {
      return
    }

    setActiveSectionId(sectionId)
    section.scrollIntoView({
      behavior: 'smooth',
      block: 'start',
    })
  }

  function handlePageNavigate(pageId: PageId) {
    setActivePageId(pageId)

    if (pageId === 'home') {
      setIsSidebarExpanded(false)
      window.scrollTo({ top: 0, behavior: 'smooth' })
      return
    }

    setActiveSectionId('input-import')
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  return {
    activePageId,
    activeSectionId,
    handleNavigate,
    handlePageNavigate,
    isSidebarExpanded,
    setIsSidebarExpanded,
  }
}
