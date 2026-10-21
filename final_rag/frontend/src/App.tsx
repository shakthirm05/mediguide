
import {
  createContext,
  useContext,
  useEffect,
  useState,
  useRef,
  type FormEvent,
  type ReactNode,
} from 'react'
import {
  BrowserRouter,
  Navigate,
  Route,
  Routes,
  useNavigate,
} from 'react-router-dom'
import {
  Activity,
  ArrowRight,
  CheckCircle2,
  Eye,
  EyeOff,
  HeartPulse,
  LockKeyhole,
  Mail,
  ShieldCheck,
  Stethoscope,
  MessageSquare,
  Plus,
  Send,
  Sparkles,
  Clock3,
  LogOut,
  Mic,
  MicOff,
  Bot,
  Database,
  LayoutDashboard,
  RefreshCw,
  Workflow,
} from 'lucide-react'
import api, { getApiError } from './lib/api'
import './App.css'

type User = {
  user_id: string
  email: string
  role: string
}

type AuthContextValue = {
  user: User | null
  loading: boolean
  login: (email: string, password: string) => Promise<void>
  register: (email: string, password: string) => Promise<void>
  logout: () => void
}

const AuthContext = createContext<AuthContextValue | null>(null)

function useAuth() {
  const context = useContext(AuthContext)

  if (!context) {
    throw new Error('useAuth must be used inside AuthProvider')
  }

  return context
}

function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let active = true

    async function validateSession() {
      const token = sessionStorage.getItem('mediguide_access_token')

      if (!token) {
        if (active) setLoading(false)
        return
      }

      try {
        const response = await api.get<User>('/api/v1/auth/me')
        if (active) setUser(response.data)
      } catch {
        sessionStorage.removeItem('mediguide_access_token')
        if (active) setUser(null)
      } finally {
        if (active) setLoading(false)
      }
    }

    void validateSession()

    return () => {
      active = false
    }
  }, [])

  async function login(email: string, password: string) {
    const response = await api.post<{
      access_token: string
      token_type: string
      expires_in: number
    }>('/api/v1/auth/login', { email, password })

    sessionStorage.setItem(
      'mediguide_access_token',
      response.data.access_token,
    )

    try {
      const account = await api.get<User>('/api/v1/auth/me')
      setUser(account.data)
    } catch (error) {
      sessionStorage.removeItem('mediguide_access_token')
      setUser(null)
      throw error
    }
  }

  async function register(email: string, password: string) {
    await api.post('/api/v1/auth/register', { email, password })
  }

  function logout() {
    sessionStorage.removeItem('mediguide_access_token')
    setUser(null)
  }

  return (
    <AuthContext.Provider
      value={{ user, loading, login, register, logout }}
    >
      {children}
    </AuthContext.Provider>
  )
}

function LoadingScreen() {
  return (
    <main className="loading-screen">
      <div className="loading-mark">
        <HeartPulse size={27} />
      </div>
      <p>Preparing your MediGuide experience…</p>
    </main>
  )
}

function ProtectedRoute({ children }: { children: ReactNode }) {
  const { user, loading } = useAuth()

  if (loading) return <LoadingScreen />
  if (!user) return <Navigate to="/login" replace />

  return <>{children}</>
}

function Brand() {
  return (
    <div className="brand">
      <div className="brand-mark">
        <HeartPulse size={25} strokeWidth={2.2} />
      </div>
      <div className="brand-copy">
        <span className="brand-name">MediGuide</span>
        <span className="brand-caption">HOSPITAL ASSISTANCE</span>
      </div>
    </div>
  )
}

function AuthPage({ mode }: { mode: 'login' | 'register' }) {
  const isRegister = mode === 'register'
  const navigate = useNavigate()
  const { login, register } = useAuth()

  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setError('')
    setNotice('')

    const normalizedEmail = email.trim()

    if (isRegister && password.length < 12) {
      setError('Your password must contain at least 12 characters.')
      return
    }

    setBusy(true)

    try {
      if (isRegister) {
        await register(normalizedEmail, password)
        setNotice('Account created successfully. Please log in.')
        navigate('/login', {
          replace: true,
          state: { registered: true },
        })
      } else {
        await login(normalizedEmail, password)
        navigate('/app', { replace: true })
      }
    } catch (requestError) {
      setError(getApiError(requestError))
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="auth-layout">
      <section className="auth-story">
        <div className="story-inner">
          <Brand />

          <div className="story-content">
            <span className="eyebrow">
              <span className="status-dot" />
              YOUR HOSPITAL COMPANION
            </span>

            <h1>
              Healthcare navigation,
              <br />
              made <span>simpler.</span>
            </h1>

            <p className="story-description">
              Get help navigating hospital services, managing appointments,
              and finding the information you need — all in one place.
            </p>

            <div className="story-features">
              <div className="story-feature">
                <div className="feature-icon">
                  <Stethoscope size={19} />
                </div>
                <div>
                  <strong>Hospital assistance</strong>
                  <p>Find departments and understand hospital processes.</p>
                </div>
              </div>

              <div className="story-feature">
                <div className="feature-icon">
                  <Activity size={19} />
                </div>
                <div>
                  <strong>Appointment management</strong>
                  <p>Access supported appointment services in one place.</p>
                </div>
              </div>

              <div className="story-feature">
                <div className="feature-icon">
                  <ShieldCheck size={19} />
                </div>
                <div>
                  <strong>Account-protected access</strong>
                  <p>Your account is validated by the MediGuide backend.</p>
                </div>
              </div>
            </div>
          </div>

          <div className="story-footer">
            <span>Designed for a more guided hospital experience.</span>
            <span className="footer-status">
              <span className="status-dot" /> Patient portal
            </span>
          </div>
        </div>
      </section>

      <section className="auth-panel">
        <div className="auth-mobile-brand">
          <Brand />
        </div>

        <div className="auth-card">
          <div className="auth-heading-icon">
            {isRegister ? (
              <HeartPulse size={24} />
            ) : (
              <LockKeyhole size={23} />
            )}
          </div>

          <p className="auth-overline">
            {isRegister ? 'GET STARTED' : 'WELCOME BACK'}
          </p>

          <h2>{isRegister ? 'Create your account' : 'Sign in to MediGuide'}</h2>

          <p className="auth-subtitle">
            {isRegister
              ? 'Create your patient account to get started.'
              : 'Enter your details to access your patient portal.'}
          </p>

          {error && (
            <div className="feedback feedback-error" role="alert">
              {error}
            </div>
          )}

          {notice && (
            <div className="feedback feedback-success" role="status">
              {notice}
            </div>
          )}

          <form className="auth-form" onSubmit={handleSubmit}>
            <label htmlFor="email">Email address</label>
            <div className="input-wrap">
              <Mail size={18} aria-hidden="true" />
              <input
                id="email"
                type="email"
                autoComplete="email"
                placeholder="you@example.com"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                maxLength={255}
                required
              />
            </div>

            <div className="password-label-row">
              <label htmlFor="password">Password</label>
              {!isRegister && (
                <span className="password-hint">Your account password</span>
              )}
            </div>

            <div className="input-wrap">
              <LockKeyhole size={18} aria-hidden="true" />
              <input
                id="password"
                type={showPassword ? 'text' : 'password'}
                autoComplete={isRegister ? 'new-password' : 'current-password'}
                placeholder={
                  isRegister ? 'At least 12 characters' : 'Enter your password'
                }
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                minLength={isRegister ? 12 : 1}
                maxLength={128}
                required
              />
              <button
                className="password-toggle"
                type="button"
                aria-label={showPassword ? 'Hide password' : 'Show password'}
                onClick={() => setShowPassword((value) => !value)}
              >
                {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
              </button>
            </div>

            {isRegister && (
              <p className="form-help">
                Use 12–128 characters. Choose a password you do not use
                elsewhere.
              </p>
            )}

            <button className="submit-button" type="submit" disabled={busy}>
              {busy
                ? 'Please wait…'
                : isRegister
                  ? 'Create account'
                  : 'Sign in'}
              {!busy && <ArrowRight size={18} />}
            </button>
          </form>

          <div className="auth-switch">
            {isRegister ? 'Already have an account?' : 'New to MediGuide?'}
            <button
              type="button"
              onClick={() =>
                navigate(isRegister ? '/login' : '/register')
              }
            >
              {isRegister ? 'Sign in' : 'Create an account'}
            </button>
          </div>

          <div className="privacy-note">
            <ShieldCheck size={17} />
            <span>
              Your session is validated by the MediGuide API. Never share your
              password or access token.
            </span>
          </div>
        </div>

        <p className="panel-footer">
          MediGuide · Hospital assistance, connected.
        </p>
      </section>
    </main>
  )
}

type PatientProfileStatus = {
  linked: boolean
  patient_identifier: string | null
}

function PatientProfileGate({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<
    'checking' | 'unlinked' | 'linked' | 'error'
  >('checking')
  const [patientIdentifier, setPatientIdentifier] = useState('')
  const [claimCode, setClaimCode] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [checkAttempt, setCheckAttempt] = useState(0)
  const { logout } = useAuth()
  const navigate = useNavigate()

  useEffect(() => {
    let active = true

    async function checkProfileStatus() {
      setStatus('checking')
      setError('')

      try {
        const response = await api.get<PatientProfileStatus>(
          '/api/v1/patient/profile-status',
        )

        if (active) {
          setStatus(response.data.linked ? 'linked' : 'unlinked')
        }
      } catch (requestError) {
        if (active) {
          setError(getApiError(requestError))
          setStatus('error')
        }
      }
    }

    void checkProfileStatus()

    return () => {
      active = false
    }
  }, [checkAttempt])

  async function handleLinkPatient(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setError('')
    setNotice('')
    setBusy(true)

    try {
      await api.post('/api/v1/patient/link', {
        patient_identifier: patientIdentifier.trim(),
        claim_code: claimCode,
      })

      setClaimCode('')
      setNotice('Patient record linked. Verifying your profile…')
      setStatus('checking')
      setCheckAttempt((attempt) => attempt + 1)
    } catch (requestError) {
      setError(getApiError(requestError))
    } finally {
      setBusy(false)
    }
  }

  function handleLogout() {
    logout()
    navigate('/login', { replace: true })
  }

  if (status === 'checking') {
    return <LoadingScreen />
  }

  if (status === 'linked') {
    return <>{children}</>
  }

  return (
    <main className="patient-link-layout">
      <header className="patient-link-header">
        <Brand />
        <button
          className="logout-button"
          type="button"
          onClick={handleLogout}
        >
          Sign out
        </button>
      </header>

      <section className="patient-link-content">
        <div className="patient-link-icon">
          <HeartPulse size={25} />
        </div>

        <p className="auth-overline">PATIENT ONBOARDING</p>
        <h1>Link your patient record</h1>
        <p className="patient-link-description">
          Connect your MediGuide account to your existing hospital patient
          record to access patient-specific services, including appointment
          management.
        </p>

        {status === 'error' ? (
          <>
            <div className="feedback feedback-error" role="alert">
              {error}
            </div>
            <button
              className="submit-button"
              type="button"
              onClick={() => setCheckAttempt((attempt) => attempt + 1)}
            >
              Retry profile check <ArrowRight size={18} />
            </button>
          </>
        ) : (
          <>
            <div className="patient-link-info">
              <ShieldCheck size={19} />
              <p>
                Use the patient identifier and one-time claim code provided
                through your hospital’s authorized registration process.
                Creating a MediGuide account alone does not create a hospital
                patient record.
              </p>
            </div>

            {error && (
              <div className="feedback feedback-error" role="alert">
                {error}
              </div>
            )}

            {notice && (
              <div className="feedback feedback-success" role="status">
                {notice}
              </div>
            )}

            <form className="auth-form" onSubmit={handleLinkPatient}>
              <label htmlFor="patient-identifier">Patient identifier</label>
              <div className="input-wrap">
                <input
                  id="patient-identifier"
                  type="text"
                  autoComplete="off"
                  placeholder="e.g. MEDI-P002"
                  value={patientIdentifier}
                  onChange={(event) =>
                    setPatientIdentifier(event.target.value)
                  }
                  minLength={1}
                  maxLength={50}
                  required
                  disabled={busy}
                />
              </div>

              <label className="patient-claim-label" htmlFor="claim-code">
                One-time claim code
              </label>
              <div className="input-wrap">
                <input
                  id="claim-code"
                  type="password"
                  autoComplete="off"
                  placeholder="Enter your claim code"
                  value={claimCode}
                  onChange={(event) => setClaimCode(event.target.value)}
                  maxLength={200}
                  required
                  disabled={busy}
                />
              </div>

              <button
                className="submit-button"
                type="submit"
                disabled={busy || !patientIdentifier.trim() || !claimCode}
              >
                {busy ? 'Linking patient record…' : 'Link patient record'}
                {!busy && <ArrowRight size={18} />}
              </button>
            </form>
          </>
        )}
      </section>

      <footer className="patient-link-footer">
        MediGuide · Hospital assistance, connected.
      </footer>
    </main>
  )
}


type Conversation = {
  conversation_id: string
  created_at: string
  updated_at: string
}

type ChatMessage = {
  role: 'user' | 'assistant'
  content: string
}

type Department = {
  department_id: string
  name: string
  location_code: string | null
  floor_label: string | null
}

type Doctor = {
  doctor_id: string
  display_name: string
  department_id: string
  department_name: string
}

type AppointmentSlot = {
  doctor_id: string
  doctor_name: string
  department_name: string
  date: string
  starts_at: string
  ends_at: string
  timezone: string
}

type Appointment = {
  appointment_id: string
  booking_reference: string
  doctor_id: string
  doctor_name?: string
  department_id?: string
  department_name?: string
  location_code?: string | null
  floor_label?: string | null
  starts_at: string
  ends_at: string
  status: string
  reason?: string | null
  cancelled_at?: string | null
}

function formatAppointmentDate(value: string) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value

  return date.toLocaleString(undefined, {
    weekday: 'short',
    day: 'numeric',
    month: 'short',
    year: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  })
}

function localDateString() {
  const now = new Date()
  const year = now.getFullYear()
  const month = String(now.getMonth() + 1).padStart(2, '0')
  const day = String(now.getDate()).padStart(2, '0')
  return `${year}-${month}-${day}`
}

function AppointmentsPanel() {
  const [appointments, setAppointments] = useState<Appointment[]>([])
  const [departments, setDepartments] = useState<Department[]>([])
  const [doctors, setDoctors] = useState<Doctor[]>([])
  const [slots, setSlots] = useState<AppointmentSlot[]>([])

  const [departmentId, setDepartmentId] = useState('')
  const [doctorId, setDoctorId] = useState('')
  const [appointmentDate, setAppointmentDate] = useState(localDateString())
  const [reason, setReason] = useState('')

  const [rescheduleReference, setRescheduleReference] = useState('')
  const [rescheduleDate, setRescheduleDate] = useState(localDateString())
  const [rescheduleSlots, setRescheduleSlots] = useState<AppointmentSlot[]>([])
  const [rescheduleSlot, setRescheduleSlot] = useState('')

  const [loading, setLoading] = useState(true)
  const [loadingDoctors, setLoadingDoctors] = useState(false)
  const [loadingSlots, setLoadingSlots] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')

  async function loadAppointments() {
    const response = await api.get<Appointment[]>(
      '/api/v1/appointments/me',
    )
    setAppointments(response.data)
  }

  useEffect(() => {
    let active = true

    async function loadInitialData() {
      setLoading(true)
      setError('')

      try {
        const [departmentResponse, appointmentResponse] =
          await Promise.all([
            api.get<Department[]>('/api/v1/directory/departments'),
            api.get<Appointment[]>('/api/v1/appointments/me'),
          ])

        if (!active) return

        setDepartments(departmentResponse.data)
        setAppointments(appointmentResponse.data)
      } catch (requestError) {
        if (active) setError(getApiError(requestError))
      } finally {
        if (active) setLoading(false)
      }
    }

    void loadInitialData()

    return () => {
      active = false
    }
  }, [])

  useEffect(() => {
    let active = true

    async function loadDoctors() {
      setDoctors([])
      setDoctorId('')
      setSlots([])

      if (!departmentId) return

      setLoadingDoctors(true)

      try {
        const response = await api.get<Doctor[]>(
          '/api/v1/directory/doctors',
          { params: { department_id: departmentId } },
        )

        if (active) setDoctors(response.data)
      } catch (requestError) {
        if (active) setError(getApiError(requestError))
      } finally {
        if (active) setLoadingDoctors(false)
      }
    }

    void loadDoctors()

    return () => {
      active = false
    }
  }, [departmentId])

  async function loadSlots(
    selectedDoctorId: string,
    selectedDate: string,
    mode: 'booking' | 'reschedule',
  ) {
    setError('')
    setNotice('')

    if (!selectedDoctorId || !selectedDate) {
      if (mode === 'booking') setSlots([])
      else setRescheduleSlots([])
      return
    }

    if (mode === 'booking') setLoadingSlots(true)
    else setLoadingSlots(true)

    try {
      const response = await api.get<AppointmentSlot[]>(
        '/api/v1/appointments/availability',
        {
          params: {
            doctor_id: selectedDoctorId,
            date: selectedDate,
          },
        },
      )

      if (mode === 'booking') setSlots(response.data)
      else setRescheduleSlots(response.data)
    } catch (requestError) {
      setError(getApiError(requestError))
      if (mode === 'booking') setSlots([])
      else setRescheduleSlots([])
    } finally {
      setLoadingSlots(false)
    }
  }

  async function bookAppointment(slot: AppointmentSlot) {
    setError('')
    setNotice('')
    setBusy(true)

    try {
      await api.post('/api/v1/appointments', {
        doctor_id: slot.doctor_id,
        starts_at: slot.starts_at,
        reason: reason.trim() || null,
      })

      setNotice('Appointment booked successfully.')
      setReason('')
      setSlots([])
      await loadAppointments()
    } catch (requestError) {
      setError(getApiError(requestError))
      // Availability can change between loading and booking.
      // Refresh the slots before the patient tries again.
      await loadSlots(doctorId, appointmentDate, 'booking')
    } finally {
      setBusy(false)
    }
  }

  async function cancelAppointment(reference: string) {
    const confirmed = window.confirm(
      `Cancel appointment ${reference}?`,
    )
    if (!confirmed) return

    setError('')
    setNotice('')
    setBusy(true)

    try {
      await api.post(
        `/api/v1/appointments/${encodeURIComponent(reference)}/cancel`,
      )
      setNotice(`Appointment ${reference} cancelled.`)
      await loadAppointments()
    } catch (requestError) {
      setError(getApiError(requestError))
    } finally {
      setBusy(false)
    }
  }

  async function submitReschedule() {
    if (!rescheduleReference || !rescheduleSlot) {
      setError('Choose an available slot before rescheduling.')
      return
    }

    setError('')
    setNotice('')
    setBusy(true)

    try {
      await api.post(
        `/api/v1/appointments/${encodeURIComponent(rescheduleReference)}/reschedule`,
        { new_starts_at: rescheduleSlot },
      )

      setNotice(`Appointment ${rescheduleReference} rescheduled.`)
      setRescheduleReference('')
      setRescheduleSlots([])
      setRescheduleSlot('')
      await loadAppointments()
    } catch (requestError) {
      setError(getApiError(requestError))
      const current = appointments.find(
        (item) => item.booking_reference === rescheduleReference,
      )
      if (current) {
        await loadSlots(current.doctor_id, rescheduleDate, 'reschedule')
      }
    } finally {
      setBusy(false)
    }
  }

  const selectedDoctor = doctors.find(
    (doctor) => doctor.doctor_id === doctorId,
  )

  return (
    <div className="appointments-panel">
      <div className="appointments-heading">
        <div>
          <p className="chat-topbar-eyebrow">PATIENT SERVICES</p>
          <h2>Appointments</h2>
          <p>Book a visit or manage your existing appointments.</p>
        </div>

        <button
          className="appointment-secondary-button"
          type="button"
          disabled={loading || busy}
          onClick={() => {
            setError('')
            setNotice('')
            void loadAppointments().catch((e) =>
              setError(getApiError(e)),
            )
          }}
        >
          Refresh
        </button>
      </div>

      {error && (
        <div className="feedback feedback-error" role="alert">
          {error}
        </div>
      )}

      {notice && (
        <div className="feedback feedback-success" role="status">
          {notice}
        </div>
      )}

      <section className="appointment-card">
        <div className="appointment-card-heading">
          <div>
            <h3>Book an appointment</h3>
            <p>Select a department, doctor, date, and available time.</p>
          </div>
        </div>

        <div className="appointment-form-grid">
          <label>
            Department
            <select
              value={departmentId}
              onChange={(event) => {
                setDepartmentId(event.target.value)
                setSlots([])
              }}
              disabled={loading || busy}
            >
              <option value="">Choose department</option>
              {departments.map((department) => (
                <option
                  key={department.department_id}
                  value={department.department_id}
                >
                  {department.name}
                </option>
              ))}
            </select>
          </label>

          <label>
            Doctor
            <select
              value={doctorId}
              onChange={(event) => {
                setDoctorId(event.target.value)
                setSlots([])
              }}
              disabled={!departmentId || loadingDoctors || busy}
            >
              <option value="">
                {loadingDoctors ? 'Loading doctors…' : 'Choose doctor'}
              </option>
              {doctors.map((doctor) => (
                <option key={doctor.doctor_id} value={doctor.doctor_id}>
                  {doctor.display_name}
                </option>
              ))}
            </select>
          </label>

          <label>
            Appointment date
            <input
              type="date"
              min={localDateString()}
              value={appointmentDate}
              onChange={(event) => {
                setAppointmentDate(event.target.value)
                setSlots([])
              }}
              disabled={busy}
            />
          </label>

          <label>
            Reason (optional)
            <input
              type="text"
              value={reason}
              maxLength={500}
              placeholder="Brief reason for your visit"
              onChange={(event) => setReason(event.target.value)}
              disabled={busy}
            />
          </label>
        </div>

        <button
          className="appointment-primary-button"
          type="button"
          disabled={!doctorId || !appointmentDate || busy}
          onClick={() => void loadSlots(doctorId, appointmentDate, 'booking')}
        >
          Find available slots
        </button>

        {selectedDoctor && (
          <p className="appointment-context">
            {selectedDoctor.display_name} · {selectedDoctor.department_name}
          </p>
        )}

        {loadingSlots && (
          <p className="appointment-muted">Loading available slots…</p>
        )}

        {!loadingSlots && slots.length > 0 && (
          <div className="appointment-slots">
            <h4>Available times</h4>
            <div className="appointment-slot-list">
              {slots.map((slot) => (
                <button
                  type="button"
                  className="appointment-slot"
                  key={slot.starts_at}
                  disabled={busy}
                  onClick={() => void bookAppointment(slot)}
                >
                  {new Date(slot.starts_at).toLocaleTimeString(undefined, {
                    hour: 'numeric',
                    minute: '2-digit',
                  })}
                  <span>Book</span>
                </button>
              ))}
            </div>
            <p className="appointment-muted">
              Times are displayed in your browser's local timezone.
              The slot's configured timezone is {slots[0]?.timezone}.
            </p>
          </div>
        )}

        {!loadingSlots &&
          doctorId &&
          slots.length === 0 && (
            <p className="appointment-muted">
              Select “Find available slots” to check availability for this date.
            </p>
          )}
      </section>

      <section className="appointment-list-section">
        <div className="appointment-list-heading">
          <h3>My appointments</h3>
          <span>{appointments.length} shown</span>
        </div>

        {loading ? (
          <p className="appointment-muted">Loading appointments…</p>
        ) : appointments.length === 0 ? (
          <div className="appointment-empty">
            <h4>No appointments found</h4>
            <p>Your upcoming appointments will appear here after booking.</p>
          </div>
        ) : (
          <div className="appointment-list">
            {appointments.map((appointment) => {
              const isActive = appointment.status === 'booked'
              const isRescheduling =
                rescheduleReference === appointment.booking_reference

              return (
                <article
                  className="appointment-item"
                  key={appointment.appointment_id}
                >
                  <div className="appointment-item-top">
                    <div>
                      <span
                        className={`appointment-status ${
                          isActive ? 'appointment-status-active' : ''
                        }`}
                      >
                        {appointment.status}
                      </span>
                      <h4>{appointment.doctor_name || 'Doctor'}</h4>
                      <p>{appointment.department_name || 'Department'}</p>
                    </div>
                    <span className="appointment-reference">
                      {appointment.booking_reference}
                    </span>
                  </div>

                  <div className="appointment-details">
                    <p>
                      <strong>When</strong>
                      {formatAppointmentDate(appointment.starts_at)}
                    </p>
                    <p>
                      <strong>Location</strong>
                      {[appointment.location_code, appointment.floor_label]
                        .filter(Boolean)
                        .join(' · ') || 'Not provided'}
                    </p>
                    {appointment.reason && (
                      <p>
                        <strong>Reason</strong>
                        {appointment.reason}
                      </p>
                    )}
                  </div>

                  {isActive && (
                    <div className="appointment-actions">
                      <button
                        className="appointment-secondary-button"
                        type="button"
                        disabled={busy}
                        onClick={() => {
                          setError('')
                          setNotice('')
                          setRescheduleReference(
                            isRescheduling
                              ? ''
                              : appointment.booking_reference,
                          )
                          setRescheduleDate(localDateString())
                          setRescheduleSlots([])
                          setRescheduleSlot('')
                        }}
                      >
                        {isRescheduling ? 'Close rescheduling' : 'Reschedule'}
                      </button>

                      <button
                        className="appointment-cancel-button"
                        type="button"
                        disabled={busy}
                        onClick={() =>
                          void cancelAppointment(appointment.booking_reference)
                        }
                      >
                        Cancel appointment
                      </button>
                    </div>
                  )}

                  {isRescheduling && (
                    <div className="reschedule-box">
                      <label>
                        New date
                        <input
                          type="date"
                          min={localDateString()}
                          value={rescheduleDate}
                          onChange={(event) => {
                            setRescheduleDate(event.target.value)
                            setRescheduleSlots([])
                            setRescheduleSlot('')
                          }}
                          disabled={busy}
                        />
                      </label>

                      <button
                        className="appointment-secondary-button"
                        type="button"
                        disabled={busy || !rescheduleDate}
                        onClick={() =>
                          void loadSlots(
                            appointment.doctor_id,
                            rescheduleDate,
                            'reschedule',
                          )
                        }
                      >
                        Find new times
                      </button>

                      {rescheduleSlots.length > 0 && (
                        <>
                          <label>
                            Available time
                            <select
                              value={rescheduleSlot}
                              onChange={(event) =>
                                setRescheduleSlot(event.target.value)
                              }
                              disabled={busy}
                            >
                              <option value="">Choose a time</option>
                              {rescheduleSlots.map((slot) => (
                                <option
                                  key={slot.starts_at}
                                  value={slot.starts_at}
                                >
                                  {new Date(slot.starts_at).toLocaleTimeString(
                                    undefined,
                                    {
                                      hour: 'numeric',
                                      minute: '2-digit',
                                    },
                                  )}
                                </option>
                              ))}
                            </select>
                          </label>

                          <button
                            className="appointment-primary-button"
                            type="button"
                            disabled={busy || !rescheduleSlot}
                            onClick={() => void submitReschedule()}
                          >
                            Confirm reschedule
                          </button>
                        </>
                      )}

                      {!loadingSlots &&
                        rescheduleSlots.length === 0 && (
                          <p className="appointment-muted">
                            Choose a date and find new times to continue.
                          </p>
                        )}
                    </div>
                  )}
                </article>
              )
            })}
          </div>
        )}
      </section>
    </div>
  )
}

function Dashboard() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  const [conversations, setConversations] = useState<Conversation[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [activeView, setActiveView] = useState<'chat' | 'appointments'>('chat')
  const [question, setQuestion] = useState('')
  const [loadingList, setLoadingList] = useState(true)
  const [loadingHistory, setLoadingHistory] = useState(false)
  const [sending, setSending] = useState(false)
  const [creating, setCreating] = useState(false)
  const [error, setError] = useState('')
  const [isRecording, setIsRecording] = useState(false)
  const [voiceProcessing, setVoiceProcessing] = useState(false)
  const [voiceAudioSrc, setVoiceAudioSrc] = useState('')
  const recorderRef = useRef<MediaRecorder | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const audioChunksRef = useRef<Blob[]>([])

  async function refreshConversations() {
    const response = await api.get<{
      items: Conversation[]
      total: number
      limit: number
      offset: number
    }>('/api/v1/conversations')

    setConversations(response.data.items)
    return response.data.items
  }

  useEffect(() => {
    let active = true

    async function loadConversations() {
      try {
        const response = await api.get<{
          items: Conversation[]
          total: number
          limit: number
          offset: number
        }>('/api/v1/conversations')

        if (active) {
          setConversations(response.data.items)
        }
      } catch (requestError) {
        if (active) setError(getApiError(requestError))
      } finally {
        if (active) setLoadingList(false)
      }
    }

    void loadConversations()

    return () => {
      active = false
    }
  }, [])

  async function openConversation(conversationId: string) {
    setActiveId(conversationId)
    setMessages([])
    setError('')
    setLoadingHistory(true)

    try {
      const response = await api.get<{
        conversation_id: string
        messages: ChatMessage[]
        count: number
      }>(
        `/api/v1/conversations/${encodeURIComponent(conversationId)}/messages`,
      )

      if (response.data.conversation_id === conversationId) {
        setMessages(response.data.messages)
      }
    } catch (requestError) {
      setError(getApiError(requestError))
    } finally {
      setLoadingHistory(false)
    }
  }

  async function createConversation() {
    setCreating(true)
    setError('')

    try {
      const response = await api.post<Conversation>(
        '/api/v1/conversations',
      )

      const created = response.data

      setConversations((current) => [
        created,
        ...current.filter(
          (item) => item.conversation_id !== created.conversation_id,
        ),
      ])

      setActiveId(created.conversation_id)
      setMessages([])
      setQuestion('')
    } catch (requestError) {
      setError(getApiError(requestError))
    } finally {
      setCreating(false)
    }
  }

  async function sendMessage(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()

    const text = question.trim()
    const conversationId = activeId

    if (!text || !conversationId || sending) return

    setQuestion('')
    setError('')
    setMessages((current) => [
      ...current,
      { role: 'user', content: text },
    ])
    setSending(true)

    try {
      const response = await api.post<{
        conversation_id: string
        answer: string
        intent: string
        workflow_status: string
        evidence: unknown[]
      }>(
        '/api/v1/chat',
        {
          question: text,
          conversation_id: conversationId,
        },
        { timeout: 120000 },
      )

      if (response.data.conversation_id !== conversationId) {
        throw new Error('The backend returned a different conversation ID.')
      }

      setMessages((current) => [
        ...current,
        { role: 'assistant', content: response.data.answer },
      ])

      try {
        await refreshConversations()
      } catch {
        // The chat response succeeded; a sidebar refresh can be retried later.
      }
    } catch (requestError) {
      setError(
        requestError instanceof Error &&
        requestError.message ===
          'The backend returned a different conversation ID.'
          ? requestError.message
          : getApiError(requestError),
      )

      // Keep the user's text in the composer so they can retry.
      setQuestion(text)
      setMessages((current) => current.slice(0, -1))
    } finally {
      setSending(false)
    }
  }
  
async function startVoiceRecording() {
  if (!activeId || sending || voiceProcessing || isRecording) {
    if (!activeId) {
      setError('Start or select a conversation before using voice chat.')
    }
    return
  }

  if (
    !navigator.mediaDevices?.getUserMedia ||
    typeof MediaRecorder === 'undefined'
  ) {
    setError('Voice recording is not supported in this browser. Use a recent browser over HTTPS or localhost.')
    return
  }

  try {
    setError('')
    setVoiceAudioSrc('')

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: true,
    })
    streamRef.current = stream

    const preferredType = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/mp4',
    ].find((type) => MediaRecorder.isTypeSupported(type))

    const recorder = preferredType
      ? new MediaRecorder(stream, { mimeType: preferredType })
      : new MediaRecorder(stream)

    audioChunksRef.current = []
    recorderRef.current = recorder

    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        audioChunksRef.current.push(event.data)
      }
    }

    recorder.onerror = () => {
      setError('Recording failed. Please check your microphone and try again.')
      stream.getTracks().forEach((track) => track.stop())
      streamRef.current = null
      setIsRecording(false)
    }

    recorder.onstop = () => {
      const mimeType = recorder.mimeType || 'audio/webm'
      const audioBlob = new Blob(audioChunksRef.current, { type: mimeType })

      stream.getTracks().forEach((track) => track.stop())
      streamRef.current = null
      recorderRef.current = null
      setIsRecording(false)

      if (audioBlob.size === 0) {
        setError('No audio was recorded. Please try again.')
        return
      }

      void submitVoiceAudio(audioBlob, mimeType)
    }

    recorder.start()
    setIsRecording(true)
  } catch {
    setError(
      'Could not access your microphone. Allow microphone permission and try again.',
    )
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    setIsRecording(false)
  }
}

function stopVoiceRecording() {
  const recorder = recorderRef.current

  if (recorder && recorder.state !== 'inactive') {
    recorder.stop()
  }
}

async function submitVoiceAudio(blob: Blob, mimeType: string) {
  const conversationId = activeId

  if (!conversationId || voiceProcessing) return

  setVoiceProcessing(true)
  setError('')

  try {
    const extension = mimeType.includes('mp4')
      ? 'mp4'
      : mimeType.includes('ogg')
        ? 'ogg'
        : 'webm'

    const formData = new FormData()
    formData.append('audio', blob, `recording.${extension}`)
    formData.append('conversation_id', conversationId)

    const response = await api.post<{
      conversation_id: string
      transcript: string
      answer: string
      audio_mime_type?: string | null
      audio_base64?: string | null
    }>('/api/v1/speech/voice-chat', formData, {
      timeout: 120000,
      headers: { 'Content-Type': 'multipart/form-data' },
    })

    if (response.data.conversation_id !== conversationId) {
      throw new Error('The backend returned a different conversation ID.')
    }

    const transcript = response.data.transcript?.trim()
    const answer = response.data.answer?.trim()

    if (!transcript || !answer) {
      throw new Error('The voice response did not contain a transcript or answer.')
    }

    setMessages((current) => [
      ...current,
      { role: 'user', content: transcript },
      { role: 'assistant', content: answer },
    ])

    if (response.data.audio_base64) {
      const mimeType = response.data.audio_mime_type || 'audio/wav'
      setVoiceAudioSrc(
        `data:${mimeType};base64,${response.data.audio_base64}`,
      )
    }

    try {
      await refreshConversations()
    } catch {
      // The voice response succeeded; refresh can be retried later.
    }
  } catch (requestError) {
    setError(
      requestError instanceof Error &&
        requestError.message ===
          'The backend returned a different conversation ID.'
        ? requestError.message
        : getApiError(requestError),
    )
  } finally {
    setVoiceProcessing(false)
  }
}

  function handleLogout() {
    logout()
    navigate('/login', { replace: true })
  }

  function formatDate(value: string) {
    const date = new Date(value)

    if (Number.isNaN(date.getTime())) return 'Conversation'

    return date.toLocaleDateString(undefined, {
      day: 'numeric',
      month: 'short',
    })
  }

  return (
    <main className="chat-app">
      <aside className="chat-sidebar">
        <div className="sidebar-brand">
          <Brand />
        </div>

        <button
          className="new-chat-button"
          onClick={() => void createConversation()}
          disabled={creating || sending}
        >
          <Plus size={18} />
          {creating ? 'Creating chat…' : 'New conversation'}
        </button>

        <div className="portal-navigation" aria-label="Patient portal">
  <button
    type="button"
    className={activeView === 'chat' ? 'portal-nav-active' : ''}
    aria-current={activeView === 'chat' ? 'page' : undefined}
    onClick={() => setActiveView('chat')}
  >
    <MessageSquare size={17} />
    Chat
  </button>

  <button
    type="button"
    className={
      activeView === 'appointments' ? 'portal-nav-active' : ''
    }
    aria-current={activeView === 'appointments' ? 'page' : undefined}
    onClick={() => setActiveView('appointments')}
  >
    <Clock3 size={17} />
    Appointments
  </button>
</div>

{activeView === 'chat' && (
  <>
    <div className="sidebar-section-title">
      <Clock3 size={15} />
      YOUR CONVERSATIONS
    </div>
    
    {/* Keep your existing conversation-list JSX here unchanged. */}
    <div className="conversation-list">
          {loadingList ? (
            <p className="sidebar-muted">Loading conversations…</p>
          ) : conversations.length === 0 ? (
            <p className="sidebar-muted">
              No conversations yet. Start a new chat to begin.
            </p>
          ) : (
            conversations.map((conversation) => (
              <button
                key={conversation.conversation_id}
                className={`conversation-item ${
                  activeId === conversation.conversation_id
                    ? 'conversation-item-active'
                    : ''
                }`}
                onClick={() =>
                  void openConversation(conversation.conversation_id)
                }
                disabled={sending || loadingHistory}
                title={conversation.conversation_id}
              >
                <MessageSquare size={17} />
                <span className="conversation-item-copy">
                  <span>Hospital assistance</span>
                  <small>{formatDate(conversation.updated_at)}</small>
                </span>
              </button>
            ))
          )}
        </div>
  </>
)}

        

        <div className="sidebar-account">
          <div className="sidebar-avatar">
            {(user?.email?.[0] || 'P').toUpperCase()}
          </div>
          <div className="sidebar-account-copy">
            <strong>{user?.email}</strong>
            <small>Patient account</small>
          </div>
          <button
            className="sidebar-logout"
            onClick={handleLogout}
            title="Sign out"
            aria-label="Sign out"
          >
            <LogOut size={17} />
          </button>
        </div>
      </aside>

      <section className="chat-main">
        <header className="chat-topbar">
          <div>
            <p className="chat-topbar-eyebrow">PATIENT ASSISTANCE</p>
            <h1>
  {activeView === 'appointments'
    ? 'Appointment management'
    : 'MediGuide Assistant'}
</h1>
          </div>
          <span className="chat-secure-label">
            <span className="status-dot" />
            Secure session
          </span>
        </header>
        
        {activeView === 'appointments' ? (
  <AppointmentsPanel />
) : (
  <div className="chat-content">
          {!activeId ? (
            <div className="chat-welcome">
              <div className="chat-welcome-icon">
                <Sparkles size={26} />
              </div>
              <p className="chat-overline">WELCOME TO MEDIGUIDE</p>
              <h2>How can we help you today?</h2>
              <p className="chat-welcome-description">
                Ask about hospital departments, hospital procedures, or
                supported appointment services.
              </p>
              <button
                className="welcome-start-button"
                onClick={() => void createConversation()}
                disabled={creating}
              >
                <Plus size={17} />
                {creating ? 'Creating conversation…' : 'Start a conversation'}
              </button>
            </div>
          ) : (
            <>
              <div className="chat-messages">
                {loadingHistory ? (
                  <p className="chat-empty-state">Loading conversation history…</p>
                ) : messages.length === 0 ? (
                  <div className="chat-empty-state">
                    <div className="assistant-avatar">
                      <HeartPulse size={21} />
                    </div>
                    <h2>What can I help you with?</h2>
                    <p>
                      Ask a hospital-related question to get started.
                    </p>
                  </div>
                ) : (
                  messages.map((message, index) => (
                    <div
                      className={`message-row message-row-${message.role}`}
                      key={`${activeId}-${index}`}
                    >
                      {message.role === 'assistant' && (
                        <div className="assistant-avatar">
                          <HeartPulse size={17} />
                        </div>
                      )}
                      <div className={`message-bubble message-${message.role}`}>
                        {message.content}
                      </div>
                    </div>
                  ))
                )}

                {sending && (
                  <div className="message-row message-row-assistant">
                    <div className="assistant-avatar">
                      <HeartPulse size={17} />
                    </div>
                    <div className="message-bubble message-assistant typing">
                      MediGuide is preparing a response…
                    </div>
                  </div>
                )}
              </div>

              {error && (
                <div className="chat-error" role="alert">
                  {error}
                </div>
              )}

              <form className="chat-composer" onSubmit={sendMessage}>
                <textarea
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  placeholder="Ask MediGuide a question…"
                  rows={2}
                  maxLength={2000}
                  disabled={sending || loadingHistory}
                  onKeyDown={(event) => {
                    if (
                      event.key === 'Enter' &&
                      !event.shiftKey &&
                      !event.nativeEvent.isComposing
                    ) {
                      event.preventDefault()
                      event.currentTarget.form?.requestSubmit()
                    }
                  }}
                />
                <div className="composer-footer">
                  <span>Enter to send · Shift + Enter for a new line</span>
                  <button
                    type="submit"
                    className="send-button"
                    disabled={!question.trim() || sending || loadingHistory}
                    aria-label="Send message"
                  >
                    <Send size={17} />
                    Send
                  </button>
                  <button
  type="button"
  className="voice-button"
  onClick={() => {
    if (isRecording) {
      stopVoiceRecording()
    } else {
      void startVoiceRecording()
    }
  }}
  disabled={sending || voiceProcessing || loadingHistory || !activeId}
  aria-label={isRecording ? 'Stop recording' : 'Start voice recording'}
>
  {isRecording ? <MicOff size={17} /> : <Mic size={17} />}
  {isRecording ? 'Stop recording' : 'Voice'}
</button>
                </div>
                {voiceProcessing && (
  <p className="voice-status" role="status">
    Processing your voice message…
  </p>
)}

{isRecording && (
  <p className="voice-status" role="status">
    Listening… Select Stop recording when you finish speaking.
  </p>
)}

{voiceAudioSrc && (
  <div className="voice-audio-player">
    <span>Listen to MediGuide's response</span>
    <audio controls src={voiceAudioSrc}>
      Your browser does not support audio playback.
    </audio>
  </div>
)}
              </form>
            </>
          )}
        </div>
        
)}
        <footer className="chat-footer">
          MediGuide provides hospital assistance, not medical diagnosis.
        </footer>
      </section>
    </main>
  )
}


type DeveloperOverview = {
  api: {
    status: string
    service: string
  }
  database: {
    status: string
  }
}

type PipelineInspection = {
  conversation_id: string
  intent: string | null
  dialogue_state: {
    last_intent?: string | null
    pending_question?: string | null
    last_follow_up?: boolean
    active_workflow?: unknown
  } | null
  active_workflow: unknown
  workflow_status: string | null
  next_nodes: string[]
  dialogue_slots: {
    filled_count: number
    slot_names: string[]
  }
  retrieval: {
    evidence_count: number
    evidence: Array<{
      id?: string
      chunk_id?: string
      source_id?: string
      score?: number
      rank?: number
    }>
  }
  checkpoint_available: boolean
}

type PatientStructuredData = {
  account: {
    user_id: string;
    email: string;
    role: string;
    is_active: boolean;
    created_at: string;
  };
  profile: {
    patient_id: string;
    patient_identifier: string | null;
    preferred_language: string | null;
    created_at: string;
  } | null;
  appointments: Array<Record<string, unknown>>;
  complaints: Array<Record<string, unknown>>;
  memory_consent: Array<Record<string, unknown>>;
  memory_items: Array<Record<string, unknown>>;
  conversations: Array<Record<string, unknown>>;
  limits: Record<string, number>;
};

type DeveloperView = 'dashboard' | 'pipeline' | 'robot' | 'patient-data'

function DeveloperPanel() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  const [view, setView] = useState<DeveloperView>('dashboard')
  const [overview, setOverview] = useState<DeveloperOverview | null>(null)
  const [loadingOverview, setLoadingOverview] = useState(true)
  const [error, setError] = useState('')
  
  const [pipelineUserId, setPipelineUserId] = useState('')
  const [pipelineConversationId, setPipelineConversationId] = useState('')
  const [pipelineData, setPipelineData] =
    useState<PipelineInspection | null>(null)
  const [loadingPipeline, setLoadingPipeline] = useState(false)
  const [pipelineError, setPipelineError] = useState('')
  
  const [patientLookupId, setPatientLookupId] = useState('');
  const [patientData, setPatientData] =
    useState<PatientStructuredData | null>(null);
  const [loadingPatientData, setLoadingPatientData] = useState(false);
  const [patientDataError, setPatientDataError] = useState('');



  async function loadOverview() {
    setLoadingOverview(true)
    setError('')

    try {
      const response = await api.get<DeveloperOverview>(
        '/api/v1/developer/overview',
      )
      setOverview(response.data)
    } catch (err) {
      setOverview(null)
      setError(getApiError(err))
    } finally {
      setLoadingOverview(false)
    }
  }

  useEffect(() => {
    void loadOverview()
  }, [])

  if (!user) {
  return <Navigate to="/login" replace />
}

if (user.role !== 'admin') {
  return <Navigate to="/app" replace />
}

  const navigation = [
    { id: 'dashboard' as const, label: 'Dashboard', icon: LayoutDashboard },
    { id: 'pipeline' as const, label: 'AI Pipeline', icon: Workflow },
    { id: 'patient-data' as const, label: 'Patient Data', icon: Database },
    { id: 'robot' as const, label: 'Robot Control', icon: Bot },
  ]

const loadPatientData = async () => {
  const userId = patientLookupId.trim();

  const uuidPattern =
    /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

  if (!uuidPattern.test(userId)) {
    setPatientData(null);
    setPatientDataError('Enter a valid user UUID.');
    return;
  }

  setLoadingPatientData(true);
  setPatientDataError('');
  setPatientData(null);

  try {
    const response = await api.get<PatientStructuredData>(
      `/api/v1/developer/patient-data/${encodeURIComponent(userId)}`
    );

    setPatientData(response.data);
  } catch (error: any) {
    setPatientDataError(
      error?.response?.data?.detail ||
      error?.message ||
      'Unable to load patient data.'
    );
  } finally {
    setLoadingPatientData(false);
  }
};

async function loadPipeline() {
  const userId = pipelineUserId.trim()
  const conversationId = pipelineConversationId.trim()

  const uuidPattern =
    /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i

  if (!uuidPattern.test(userId) || !uuidPattern.test(conversationId)) {
    setPipelineError(
      'Enter a valid patient user UUID and conversation UUID.',
    )
    setPipelineData(null)
    return
  }

  setLoadingPipeline(true)
  setPipelineError('')
  setPipelineData(null)

  try {
    const response = await api.get<PipelineInspection>(
      `/api/v1/developer/pipeline/${encodeURIComponent(userId)}/${encodeURIComponent(conversationId)}`,
    )

    setPipelineData(response.data)
  } catch (err) {
    setPipelineError(getApiError(err))
  } finally {
    setLoadingPipeline(false)
  }
}

  return (
    <main className="developer-layout">
      <aside className="developer-sidebar">
        <div className="developer-brand">
          <div className="developer-brand-icon">
            <HeartPulse size={24} />
          </div>
          <div>
            <strong>MediGuide</strong>
            <span>DEVELOPER PANEL</span>
          </div>
        </div>

        <div className="developer-section-label">WORKSPACE</div>

        <nav className="developer-nav" aria-label="Developer navigation">
          {navigation.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              type="button"
              className={view === id ? 'developer-nav-active' : ''}
              onClick={() => setView(id)}
            >
              <Icon size={18} />
              <span>{label}</span>
            </button>
          ))}
        </nav>

        <div className="developer-sidebar-bottom">
          <div className="developer-account">
            <ShieldCheck size={18} />
            <div>
              <strong>Developer access</strong>
              <span>{user.email}</span>
            </div>
          </div>

          <button
            type="button"
            className="developer-back-button"
            onClick={() => navigate('/app')}
          >
            Return to patient portal
          </button>

          <button
            type="button"
            className="developer-logout-button"
            onClick={() => {
              logout()
              navigate('/login', { replace: true })
            }}
          >
            <LogOut size={17} />
            Sign out
          </button>
        </div>
      </aside>

      <section className="developer-main">
        <header className="developer-topbar">
          <div>
            <span className="developer-eyebrow">MEDIGUIDE / DEVELOPER</span>
            <h1>
              {view === 'dashboard'
                ? 'Developer Dashboard'
                : view === 'pipeline'
                  ? 'AI Pipeline'
                  : view === 'patient-data'
                    ? 'Patient Structured Data'
                    : 'Robot Control'}
            </h1>
            <p>
              {view === 'dashboard'
                ? 'Monitor the actual status of your application services.'
                : view === 'pipeline'
                  ? 'Inspect the existing conversational workflow.'
                  : view === 'patient-data'
                    ? 'Look up structured records for an authorized user.'
                    : 'Robot integration and hardware status.'}
            </p>
          </div>

          {view === 'dashboard' && (
            <button
              type="button"
              className="developer-refresh-button"
              onClick={() => void loadOverview()}
              disabled={loadingOverview}
            >
              <RefreshCw size={16} />
              Refresh status
            </button>
          )}
        </header>

        {view === 'dashboard' && (
          <div className="developer-content">
            {error && (
              <div className="developer-error" role="alert">
                {error}
              </div>
            )}

            <div className="developer-status-grid">
              <article className="developer-status-card">
                <div className="developer-card-heading">
                  <span>Backend API</span>
                  <Activity size={19} />
                </div>
                <strong>
                  {loadingOverview
                    ? 'Checking…'
                    : overview?.api.status === 'ok'
                      ? 'Operational'
                      : 'Unavailable'}
                </strong>
                <p>
                  {overview?.api.service ?? 'mediguide-api'}
                </p>
                <span
                  className={`developer-status-tag ${
                    overview?.api.status === 'ok' ? 'status-good' : 'status-unknown'
                  }`}
                >
                  {loadingOverview
                    ? 'CHECKING'
                    : overview?.api.status === 'ok'
                      ? 'HEALTHY'
                      : 'NOT CONFIRMED'}
                </span>
              </article>

              <article className="developer-status-card">
                <div className="developer-card-heading">
                  <span>PostgreSQL</span>
                  <Database size={19} />
                </div>
                <strong>
                  {loadingOverview
                    ? 'Checking…'
                    : overview?.database.status === 'connected'
                      ? 'Connected'
                      : 'Unavailable'}
                </strong>
                <p>Application database connectivity</p>
                <span
                  className={`developer-status-tag ${
                    overview?.database.status === 'connected'
                      ? 'status-good'
                      : 'status-unknown'
                  }`}
                >
                  {loadingOverview
                    ? 'CHECKING'
                    : overview?.database.status === 'connected'
                      ? 'CONNECTED'
                      : 'NOT CONFIRMED'}
                </span>
              </article>
            </div>

            <section className="developer-info-panel">
              <div>
                <h2>System overview</h2>
                <p>
                  Status information is retrieved from the protected
                  Developer API. A failed request is not treated as a
                  healthy service.
                </p>
              </div>
              <div className="developer-info-row">
                <span>API endpoint</span>
                <code>GET /api/v1/developer/overview</code>
              </div>
              <div className="developer-info-row">
                <span>Graph checkpointing</span>
                <strong>PostgreSQL-backed</strong>
              </div>
              <div className="developer-info-row">
                <span>Robot integration</span>
                <strong>Not verified in this dashboard</strong>
              </div>
            </section>
          </div>
        )}

        
{view === 'pipeline' && (
  <div className="developer-content">
    <section className="developer-info-panel">
      <h2>Live checkpoint inspector</h2>
      <p>
        Enter the patient and conversation UUIDs to inspect saved
        LangGraph metadata. Message contents and private slot values
        are not displayed.
      </p>

      <form
        onSubmit={(event) => {
          event.preventDefault()
          void loadPipeline()
        }}
        className="pipeline-inspector-form"
      >
        <label htmlFor="pipeline-user-id">Patient user ID</label>
        <input
          id="pipeline-user-id"
          value={pipelineUserId}
          onChange={(event) => setPipelineUserId(event.target.value)}
          placeholder="Patient UUID"
          required
        />

        <label htmlFor="pipeline-conversation-id">
          Conversation ID
        </label>
        <input
          id="pipeline-conversation-id"
          value={pipelineConversationId}
          onChange={(event) =>
            setPipelineConversationId(event.target.value)
          }
          placeholder="Conversation UUID"
          required
        />

        <button type="submit" disabled={loadingPipeline}>
          {loadingPipeline ? 'Inspecting…' : 'Inspect checkpoint'}
        </button>
      </form>

      {pipelineError && (
        <div className="developer-error" role="alert">
          {pipelineError}
        </div>
      )}

      {pipelineData && (
        <div className="pipeline-results">
          <div className="developer-status-grid">
            <article className="developer-status-card">
              <span>Intent</span>
              <strong>{pipelineData.intent ?? 'Not recorded'}</strong>
            </article>

            <article className="developer-status-card">
              <span>Workflow status</span>
              <strong>
                {pipelineData.workflow_status ?? 'Not recorded'}
              </strong>
            </article>

            <article className="developer-status-card">
              <span>Retrieval evidence</span>
              <strong>{pipelineData.retrieval.evidence_count}</strong>
            </article>

            <article className="developer-status-card">
              <span>Filled dialogue slots</span>
              <strong>{pipelineData.dialogue_slots.filled_count}</strong>
            </article>
          </div>

          <h3>Checkpoint details</h3>
          <div className="developer-info-row">
            <span>Checkpoint available</span>
            <strong>
              {pipelineData.checkpoint_available ? 'Yes' : 'No'}
            </strong>
          </div>

          <div className="developer-info-row">
            <span>Next graph nodes</span>
            <strong>
              {pipelineData.next_nodes.length
                ? pipelineData.next_nodes.join(', ')
                : 'None'}
            </strong>
          </div>

          <div className="developer-info-row">
            <span>Filled slot names</span>
            <strong>
              {pipelineData.dialogue_slots.slot_names.length
                ? pipelineData.dialogue_slots.slot_names.join(', ')
                : 'None'}
            </strong>
          </div>

          <h3>Retrieval evidence</h3>

          {pipelineData.retrieval.evidence.length === 0 ? (
            <p>No evidence metadata was recorded in this checkpoint.</p>
          ) : (
            <div className="pipeline-evidence-list">
              {pipelineData.retrieval.evidence.map((item, index) => (
                <div
                  className="developer-info-row"
                  key={`${item.chunk_id ?? item.id ?? 'evidence'}-${index}`}
                >
                  <span>{item.chunk_id ?? item.id ?? `Evidence ${index + 1}`}</span>
                  <strong>
                    {typeof item.score === 'number'
                      ? `Score: ${item.score.toFixed(4)}`
                      : 'Score unavailable'}
                  </strong>
                </div>
              ))}
            </div>
          )}

          <p className="pipeline-disclaimer">
            This view displays the saved checkpoint, not a live
            execution trace. Retrieval scores alone do not establish
            answer quality or relevance.
          </p>
        </div>
      )}
    </section>
  </div>
)}

{view === 'patient-data' && (
  <section className="patient-data-panel">
    <div className="patient-data-intro">
      <h2>Patient Structured Data</h2>
      <p>
        Enter a user's UUID to inspect their account, profile,
        appointments, complaints, consent, saved memories, and
        conversation records.
      </p>
    </div>

    <form
      className="patient-lookup-form"
      onSubmit={(event) => {
        event.preventDefault();
        void loadPatientData();
      }}
    >
      <label htmlFor="patient-user-id">User UUID</label>

      <div className="patient-lookup-controls">
        <input
          id="patient-user-id"
          type="text"
          value={patientLookupId}
          onChange={(event) => setPatientLookupId(event.target.value)}
          placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
          autoComplete="off"
          required
        />

        <button type="submit" disabled={loadingPatientData}>
          {loadingPatientData ? 'Loading...' : 'Load Records'}
        </button>
      </div>
    </form>

    {patientDataError && (
      <div className="patient-data-error" role="alert">
        {patientDataError}
      </div>
    )}

    {loadingPatientData && (
      <p className="patient-data-status">Loading structured records...</p>
    )}

    {patientData && (
      <div className="patient-data-results">
        <div className="patient-account-card">
          <h3>Account</h3>
          <p><strong>Email:</strong> {patientData.account.email}</p>
          <p><strong>User ID:</strong> {patientData.account.user_id}</p>
          <p><strong>Role:</strong> {patientData.account.role}</p>
          <p>
            <strong>Status:</strong>{' '}
            {patientData.account.is_active ? 'Active' : 'Inactive'}
          </p>
          <p>
            <strong>Created:</strong>{' '}
            {patientData.account.created_at
              ? new Date(patientData.account.created_at).toLocaleString()
              : '—'}
          </p>
        </div>

        <div className="patient-record-grid">
          <div className="patient-record-card">
            <h3>Patient Profile</h3>
            <pre>
              {JSON.stringify(patientData.profile, null, 2)}
            </pre>
          </div>

          <div className="patient-record-card">
            <h3>Appointments ({patientData.appointments.length})</h3>
            <pre>
              {JSON.stringify(patientData.appointments, null, 2)}
            </pre>
          </div>

          <div className="patient-record-card">
            <h3>Complaints ({patientData.complaints.length})</h3>
            <pre>
              {JSON.stringify(patientData.complaints, null, 2)}
            </pre>
          </div>

          <div className="patient-record-card">
            <h3>Memory Consent</h3>
            <pre>
              {JSON.stringify(patientData.memory_consent, null, 2)}
            </pre>
          </div>

          <div className="patient-record-card">
            <h3>Stored Memory Items ({patientData.memory_items.length})</h3>
            <pre>
              {JSON.stringify(patientData.memory_items, null, 2)}
            </pre>
          </div>

          <div className="patient-record-card">
            <h3>Conversations ({patientData.conversations.length})</h3>
            <pre>
              {JSON.stringify(patientData.conversations, null, 2)}
            </pre>
          </div>
        </div>

        <p className="patient-data-disclaimer">
          Read-only inspection. The backend limits returned records;
          this view does not create, update, or delete patient data.
          Patient descriptions and saved memories may contain sensitive
          information. Access only records you are authorized to inspect.
        </p>
      </div>
    )}
  </section>
)}


        {view === 'robot' && (
          <div className="developer-content">
            <section className="developer-info-panel">
              <div className="developer-robot-heading">
                <Bot size={28} />
                <div>
                  <h2>Robot integration</h2>
                  <p>
                    Hardware connection status has not yet been
                    connected to this dashboard.
                  </p>
                </div>
              </div>

              <div className="developer-pending-note">
                <Activity size={20} />
                <div>
                  <strong>Hardware status unavailable</strong>
                  <p>
                    No connection test or navigation command is
                    issued by this screen. Real robot controls will
                    be added when the existing ROS2 interface is
                    verified.
                  </p>
                </div>
              </div>
            </section>
          </div>
        )}
      </section>
    </main>
  )
}

function AppRoutes() {
  const { user, loading } = useAuth()

  if (loading) return <LoadingScreen />

  return (
    <Routes>
      <Route
  path="/"
  element={
    <Navigate
      to={
        !user
          ? '/login'
          : user.role === 'admin'
            ? '/developer'
            : '/app'
      }
      replace
    />
  }
/>
      <Route
  path="/login"
  element={
    user ? (
      <Navigate
        to={user.role === 'admin' ? '/developer' : '/app'}
        replace
      />
    ) : (
      <AuthPage mode="login" />
    )
  }
/>
      <Route
        path="/register"
        element={
          user ? <Navigate to="/app" replace /> : <AuthPage mode="register" />
        }
      />
      
<Route
  path="/app"
  element={
    <ProtectedRoute>
      <PatientProfileGate>
        <Dashboard />
      </PatientProfileGate>
    </ProtectedRoute>
  }
/>
      <Route
  path="/developer"
  element={
    <ProtectedRoute>
      <DeveloperPanel />
    </ProtectedRoute>
  }
/>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <AppRoutes />
      </AuthProvider>
    </BrowserRouter>
  )
}
